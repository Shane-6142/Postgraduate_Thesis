import torch, os, wandb
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

dataset_save_dir = # r""
model_save_dir = # r""
os.makedirs(model_save_dir, exist_ok=True)
model_path = os.path.join(model_save_dir, "final_action_model.pth")

WANDB_PROJECT = "final-action-training"

EPOCHS = 100
BATCH_SIZE = 32
LR = 2e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
PATIENCE = 10

def load(name):
    return np.load(os.path.join(dataset_save_dir, f"{name}.npy"))

class ActionHead(nn.Module):
    def __init__(self, heatmap_dim=100, pos_dim=2, num_classes=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(heatmap_dim + pos_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, heatmap, pos_feat):
        heatmap_flat = torch.flatten(heatmap, start_dim=1)
        x = torch.cat([heatmap_flat, pos_feat], dim=-1)
        return self.net(x)
    
def upload_history_to_wandb(history, best_val_acc, best_val_loss):
    if len(history) == 0:
        print("No history to upload.")
        return

    with wandb.init(
        project=WANDB_PROJECT,
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "patience": PATIENCE,
            "input": "predicted_heatmap + current_position",
        }
    ) as run:

        for item in history:
            run.log(
                {
                    "train/loss": item["train_loss"],
                    "train/acc": item["train_acc"],
                    "val/loss": item["val_loss"],
                    "val/acc": item["val_acc"],
                },
                step=item["epoch"]
            )

        run.summary["best_val_acc"] = best_val_acc
        run.summary["best_val_loss_at_best_acc"] = best_val_loss
        run.summary["best_model_path"] = model_path
        run.summary["total_epochs_trained"] = len(history)
    
X_train_heatmap = load("X_train_action_heatmap_pred")
X_train_pos = load("X_train_action_pos")
Y_train_action = load("Y_train_action")

X_validate_heatmap = load("X_validate_action_heatmap_pred")
X_validate_pos = load("X_validate_action_pos")
Y_validate_action = load("Y_validate_action")

train_dataset = TensorDataset(
    torch.tensor(X_train_heatmap, dtype=torch.float32),
    torch.tensor(X_train_pos, dtype=torch.float32),
    torch.tensor(Y_train_action, dtype=torch.long)
)

validate_dataset = TensorDataset(
    torch.tensor(X_validate_heatmap, dtype=torch.float32),
    torch.tensor(X_validate_pos, dtype=torch.float32),
    torch.tensor(Y_validate_action, dtype=torch.long)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validate_loader = DataLoader(validate_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = ActionHead()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best_val_acc = 0.0
best_val_loss_at_best_acc = float("inf")
early_stop_counter = 0

history = []

for epoch in range(EPOCHS):
    # train
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for batch_heatmap, batch_pos, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_heatmap, batch_pos)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()
        batch_size = batch_heatmap.size(0)

        train_loss_sum += loss.item() * batch_heatmap.size(0)

        pred = torch.argmax(logits, dim=1)
        train_correct += (pred == batch_y).sum().item()
        train_total += batch_size

    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

    # validate
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch_heatmap, batch_pos, batch_y in validate_loader:
            logits = model(batch_heatmap, batch_pos)
            loss = criterion(logits, batch_y)
            batch_size = batch_heatmap.size(0)

            val_loss_sum += loss.item() * batch_size

            pred = torch.argmax(logits, dim=1)
            val_correct += (pred == batch_y).sum().item()
            val_total += batch_size

            all_val_preds.extend(pred.numpy().tolist())
            all_val_labels.extend(batch_y.numpy().tolist())

    val_loss = val_loss_sum / val_total
    val_acc = val_correct / val_total

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    })

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_loss_at_best_acc = val_loss
        early_stop_counter = 0
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "best_val_loss_at_best_acc": best_val_loss_at_best_acc,
                "epoch": epoch + 1,
                "config": {
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "max_grad_norm": MAX_GRAD_NORM,
                    "patience": PATIENCE,
                    "input": "predicted_heatmap + current_position"
                }
            },
            model_path
        )

        print("Best action model saved")

    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

print("Action head training finished.")

upload_history_to_wandb(
    history=history,
    best_val_acc=best_val_acc,
    best_val_loss=best_val_loss_at_best_acc
)

print("WandB visualization upload finished.")
