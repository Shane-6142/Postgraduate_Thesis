import torch, os, wandb
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

dataset_save_dir = # r""
model_save_dir = # r""
os.makedirs(model_save_dir, exist_ok=True)
model_path = os.path.join(model_save_dir, "final_heatmap_model.pth")

WANDB_PROJECT = "final-heatmap-training"

EPOCHS = 200
BATCH_SIZE = 32
LR = 5e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
PATIENCE = 15

class HeatmapHead(nn.Module):
    def __init__(self, text_dim=512, text_proj_dim=64):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, text_proj_dim),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, map_feat, text_feat):
        text_map = self.text_proj(text_feat)
        text_map = text_map.unsqueeze(-1).unsqueeze(-1)
        text_map = text_map.expand(-1, -1, 10, 10)

        fused = torch.cat([map_feat, text_map], dim=1)
        heatmap = self.fusion(fused)
        return heatmap.squeeze(1)

def load(name):
    return np.load(os.path.join(dataset_save_dir, f"{name}.npy"))

def heatmap_iou(pred_heatmap, gt_heatmap, threshold=0.7):
    pred_heatmap = pred_heatmap.detach()
    gt_heatmap = gt_heatmap.detach()
    pred_heatmap = torch.clamp(pred_heatmap, 0.0, 1.0)
    gt_heatmap = torch.clamp(gt_heatmap, 0.0, 1.0)
    pred_mask = pred_heatmap >= threshold
    gt_mask = gt_heatmap >= threshold
    intersection = (pred_mask & gt_mask).float().sum(dim=(1, 2))
    union = (pred_mask | gt_mask).float().sum(dim=(1, 2))
    iou = intersection / union
    return iou.mean().item()

def upload_history_to_wandb(history, best_mean_iou, best_val_loss):
    with wandb.init(
        project=WANDB_PROJECT,
        config={
            "save_dir": model_save_dir,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "text_proj_dim": 64,
            "model": "HeatmapHead",
            "save_metric": "mean_iou"
        }
    ) as run:

        for item in history:
            run.log(
                {
                    "train/loss": item["train_loss"],
                    "val/loss": item["val_loss"],
                    "val/mean_iou": item["mean_iou"],
                },
                step=item["epoch"]
            )

        run.summary["best_mean_iou"] = best_mean_iou
        run.summary["best_val_loss_at_best_iou"] = best_val_loss
        run.summary["total_epochs_trained"] = len(history)
        run.summary["best_model_path"] = model_save_dir

X_train_map = load("X_train_map_feat")
X_train_text = load("X_train_text")
Y_train = load("Y_train_heatmap")

X_validate_map = load("X_validate_map_feat")
X_validate_text = load("X_validate_text")
Y_validate = load("Y_validate_heatmap")

train_dataset = TensorDataset(
    torch.tensor(X_train_map, dtype=torch.float32),
    torch.tensor(X_train_text, dtype=torch.float32),
    torch.tensor(Y_train, dtype=torch.float32)
)

validate_dataset = TensorDataset(
    torch.tensor(X_validate_map, dtype=torch.float32),
    torch.tensor(X_validate_text, dtype=torch.float32),
    torch.tensor(Y_validate, dtype=torch.float32)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validate_loader = DataLoader(validate_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = HeatmapHead()
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best_val_loss = float("inf")
best_mean_iou = 0.0
early_stop_counter = 0
history = []

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_total = 0

    for batch_map, batch_text, batch_y in train_loader:
        optimizer.zero_grad()
        pred_heatmap = model(batch_map, batch_text)
        loss = criterion(pred_heatmap, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()
        batch_size = batch_map.size(0)

        train_loss_sum += loss.item() * batch_size
        train_total += batch_size

    train_loss = train_loss_sum / train_total

    model.eval()
    val_loss_sum = 0.0
    val_total = 0
    iou_sum = 0.0

    with torch.no_grad():
        for batch_map, batch_text, batch_y in validate_loader:
            pred_heatmap = model(batch_map, batch_text)
            loss = criterion(pred_heatmap, batch_y)
            batch_size = batch_map.size(0)
            
            val_loss_sum += loss.item() * batch_size
            val_total += batch_size

            batch_iou = heatmap_iou(pred_heatmap, batch_y, threshold=0.5)

            iou_sum += batch_iou * batch_size

    val_loss = val_loss_sum / val_total
    mean_iou = iou_sum / val_total

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "mean_iou": mean_iou
    })

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Mean IoU: {mean_iou:.6f}"
    )

    if mean_iou > best_mean_iou:
        best_mean_iou = mean_iou
        best_val_loss = val_loss
        early_stop_counter = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_mean_iou": best_mean_iou,
                "best_val_loss": best_val_loss,
                "config": {
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "max_grad_norm": MAX_GRAD_NORM,
                }
            },
            model_path
        )

        print(
            f"Best model saved | "
            f"Best Mean IoU: {best_mean_iou:.6f} | "
            f"Val Loss: {best_val_loss:.6f}"
        )
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

print("Training finished.")

upload_history_to_wandb(
    history=history,
    best_mean_iou=best_mean_iou,
    best_val_loss=best_val_loss,
)

print("WandB visualization upload finished.")
