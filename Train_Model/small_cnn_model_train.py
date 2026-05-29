import torch
import torch.nn as nn
import cv2, os, re, json, wandb
import numpy as np
from torch.utils.data import Dataset, DataLoader

train_maps_save_dir = # r""
validate_maps_save_dir = # r""
model_save_dir = # r""
os.makedirs(model_save_dir, exist_ok=True)

IMAGE_SIZE = 80
GRID_SIZE = 10
BATCH_SIZE = 16
EPOCHS = 100
LR = 5e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
PATIENCE = 10
WANDB_PROJECT = "final-mapcnn-training"

OBJECT2IDX = {
    "empty": 0,
    "obstacle": 1,
    "start": 2,
    "exit": 3,
    "cube": 4,
    "key": 5
}
IDX2OBJECT = {v: k for k, v in OBJECT2IDX.items()}

COLOR2IDX = {
    "none": 0,
    "red": 1,
    "blue": 2,
    "green": 3,
    "yellow": 4,
    "cyan": 5
}
IDX2COLOR = {v: k for k, v in COLOR2IDX.items()}

class MapCNN(nn.Module):
    def __init__(self, num_objects=6, num_colors=6):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.object_head = nn.Conv2d(64, num_objects, kernel_size=1)
        self.color_head = nn.Conv2d(64, num_colors, kernel_size=1)

    def forward(self, x):
        feat = self.conv(x) 
        object_logits = self.object_head(feat)
        color_logits = self.color_head(feat)

        return feat, object_logits, color_logits

    
def load_map_image(image_path, image_size=80):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img.astype(np.float32)

def build_labels(meta):
    object_map = np.full((GRID_SIZE, GRID_SIZE), OBJECT2IDX["empty"], dtype=np.int64)
    color_map = np.full((GRID_SIZE, GRID_SIZE), COLOR2IDX["none"], dtype=np.int64)

    # obstacles
    for x, y in meta.get("obstacles", []):
        object_map[y, x] = OBJECT2IDX["obstacle"]
        color_map[y, x] = COLOR2IDX["none"]

    # start
    sx, sy = meta["start"]
    object_map[sy, sx] = OBJECT2IDX["start"]
    color_map[sy, sx] = COLOR2IDX["none"]

    # exit
    ex, ey = meta["exit"]
    object_map[ey, ex] = OBJECT2IDX["exit"]
    color_map[ey, ex] = COLOR2IDX["none"]

    # cube
    if "cube" in meta:
        cx, cy = meta["cube"]["position"]
        cube_color = meta["cube"]["color_name"]
        object_map[cy, cx] = OBJECT2IDX["cube"]
        color_map[cy, cx] = COLOR2IDX[cube_color]

    # key
    if "key" in meta:
        kx, ky = meta["key"]["position"]
        key_color = meta["key"]["color_name"]
        object_map[ky, kx] = OBJECT2IDX["key"]
        color_map[ky, kx] = COLOR2IDX[key_color]

    return object_map, color_map

class MapDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, json_path = self.items[idx]

        image = load_map_image(image_path)
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        object_map, color_map = build_labels(meta)

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(object_map, dtype=torch.long),
            torch.tensor(color_map, dtype=torch.long)
        )

def collect_items(root_dir):
    json_files = sorted([
        f for f in os.listdir(root_dir)
        if f.startswith("map_") and f.endswith(".json")
    ])
    items = []
    for json_name in json_files:
        json_path = os.path.join(root_dir, json_name)
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        image_name = meta.get("image_name", json_name.replace(".json", ".png"))
        image_path = os.path.join(root_dir, image_name)
        nums = re.findall(r"\d+", image_name)
        map_id = int(nums[0]) if nums else len(items) + 1
        items.append((map_id, image_path, json_path))
    items = sorted(items, key=lambda x: x[0])
    items = [(image_path, json_path) for _, image_path, json_path in items]
    return items

def upload_to_wandb(history, best_score):
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
            "patience": PATIENCE
        }
    ) as run:

        for item in history:
            run.log({
                "train/loss": item["train_loss"],
                "train/object_loss": item["train_object_loss"],
                "train/color_loss": item["train_color_loss"],
                "train/object_acc": item["train_object_acc"],
                "train/color_acc": item["train_color_acc"],
                "train/non_empty_acc": item["train_non_empty_acc"],
                "train/color_obj_acc": item["train_color_obj_acc"],

                "val/loss": item["val_loss"],
                "val/object_loss": item["val_object_loss"],
                "val/color_loss": item["val_color_loss"],
                "val/object_acc": item["val_object_acc"],
                "val/color_acc": item["val_color_acc"],
                "val/non_empty_acc": item["val_non_empty_acc"],
                "val/color_obj_acc": item["val_color_obj_acc"],

                "val/score": item["val_score"]
            }, step=item["epoch"])

        run.summary["best_score"] = best_score


if __name__ == "__main__":
    train_items = collect_items(train_maps_save_dir)
    val_items = collect_items(validate_maps_save_dir)

    if len(train_items) == 0:
        raise RuntimeError("No train map json/image pairs found.")

    if len(val_items) == 0:
        raise RuntimeError("No validation map json/image pairs found.")

    print("Train maps:", len(train_items))
    print("Val maps:", len(val_items))
    print("Total maps:", len(train_items) + len(val_items))

    train_loader = DataLoader(
        MapDataset(train_items),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        MapDataset(val_items),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = MapCNN(
        num_objects=len(OBJECT2IDX),
        num_colors=len(COLOR2IDX)
    )

    object_weights = torch.tensor(
        [0.2, 1.0, 2.0, 2.0, 2.0, 2.0],
        dtype=torch.float32
    )
    color_weights = torch.tensor(
        [0.2, 1.5, 1.5, 1.5, 1.5, 1.5],
        dtype=torch.float32
    )

    criterion_object = nn.CrossEntropyLoss(weight=object_weights)
    criterion_color = nn.CrossEntropyLoss(weight=color_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model_path = os.path.join(model_save_dir, "mapcnn_feat_model.pth")

    best_score = -1.0
    early_stop_counter = 0
    history = []

    for epoch in range(EPOCHS):
        # Train
        model.train()

        train_loss_sum = 0.0
        train_object_loss_sum = 0.0
        train_color_loss_sum = 0.0

        train_object_acc_sum = 0.0
        train_color_acc_sum = 0.0
        train_non_empty_acc_sum = 0.0
        train_color_obj_acc_sum = 0.0

        train_total = 0

        for image, object_map, color_map in train_loader:
            batch_size = image.size(0)
            optimizer.zero_grad()
            feat, object_logits, color_logits = model(image)
            loss_object = criterion_object(object_logits, object_map)
            loss_color = criterion_color(color_logits, color_map)
            loss = loss_object + 0.5 * loss_color
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=MAX_GRAD_NORM
            )
            optimizer.step()

            with torch.no_grad():
                object_pred = torch.argmax(object_logits, dim=1)
                color_pred = torch.argmax(color_logits, dim=1)

                object_acc = (object_pred == object_map).float().mean()
                color_acc = (color_pred == color_map).float().mean()

                non_empty_mask = object_map != OBJECT2IDX["empty"]
                if non_empty_mask.sum() > 0:
                    non_empty_acc = (
                        object_pred[non_empty_mask]
                        == object_map[non_empty_mask]
                    ).float().mean()
                else:
                    non_empty_acc = torch.tensor(0.0)

                color_obj_mask = color_map != COLOR2IDX["none"]
                if color_obj_mask.sum() > 0:
                    color_obj_acc = (
                        color_pred[color_obj_mask]
                        == color_map[color_obj_mask]
                    ).float().mean()
                else:
                    color_obj_acc = torch.tensor(0.0)

            train_loss_sum += loss.item() * batch_size
            train_object_loss_sum += loss_object.item() * batch_size
            train_color_loss_sum += loss_color.item() * batch_size

            train_object_acc_sum += object_acc.item() * batch_size
            train_color_acc_sum += color_acc.item() * batch_size
            train_non_empty_acc_sum += non_empty_acc.item() * batch_size
            train_color_obj_acc_sum += color_obj_acc.item() * batch_size

            train_total += batch_size

        train_loss = train_loss_sum / train_total
        train_object_loss = train_object_loss_sum / train_total
        train_color_loss = train_color_loss_sum / train_total
        train_object_acc = train_object_acc_sum / train_total
        train_color_acc = train_color_acc_sum / train_total
        train_non_empty_acc = train_non_empty_acc_sum / train_total
        train_color_obj_acc = train_color_obj_acc_sum / train_total

        # Validate
        model.eval()

        val_loss_sum = 0.0
        val_object_loss_sum = 0.0
        val_color_loss_sum = 0.0

        val_object_acc_sum = 0.0
        val_color_acc_sum = 0.0
        val_non_empty_acc_sum = 0.0
        val_color_obj_acc_sum = 0.0

        val_total = 0

        with torch.no_grad():
            for image, object_map, color_map in val_loader:
                batch_size = image.size(0)
                feat, object_logits, color_logits = model(image)
                loss_object = criterion_object(object_logits, object_map)
                loss_color = criterion_color(color_logits, color_map)
                loss = loss_object + 0.5 * loss_color
                object_pred = torch.argmax(object_logits, dim=1)
                color_pred = torch.argmax(color_logits, dim=1)
                object_acc = (object_pred == object_map).float().mean()
                color_acc = (color_pred == color_map).float().mean()
                non_empty_mask = object_map != OBJECT2IDX["empty"]
                if non_empty_mask.sum() > 0:
                    non_empty_acc = (
                        object_pred[non_empty_mask]
                        == object_map[non_empty_mask]
                    ).float().mean()
                else:
                    non_empty_acc = torch.tensor(0.0)

                color_obj_mask = color_map != COLOR2IDX["none"]
                if color_obj_mask.sum() > 0:
                    color_obj_acc = (
                        color_pred[color_obj_mask]
                        == color_map[color_obj_mask]
                    ).float().mean()
                else:
                    color_obj_acc = torch.tensor(0.0)

                val_loss_sum += loss.item() * batch_size
                val_object_loss_sum += loss_object.item() * batch_size
                val_color_loss_sum += loss_color.item() * batch_size

                val_object_acc_sum += object_acc.item() * batch_size
                val_color_acc_sum += color_acc.item() * batch_size
                val_non_empty_acc_sum += non_empty_acc.item() * batch_size
                val_color_obj_acc_sum += color_obj_acc.item() * batch_size

                val_total += batch_size

        val_loss = val_loss_sum / val_total
        val_object_loss = val_object_loss_sum / val_total
        val_color_loss = val_color_loss_sum / val_total
        val_object_acc = val_object_acc_sum / val_total
        val_color_acc = val_color_acc_sum / val_total
        val_non_empty_acc = val_non_empty_acc_sum / val_total
        val_color_obj_acc = val_color_obj_acc_sum / val_total

        val_score = (
            0.5 * val_non_empty_acc +
            0.3 * val_color_obj_acc +
            0.2 * val_object_acc
        )

        history.append({
            "epoch": epoch + 1,

            "train_loss": train_loss,
            "train_object_loss": train_object_loss,
            "train_color_loss": train_color_loss,
            "train_object_acc": train_object_acc,
            "train_color_acc": train_color_acc,
            "train_non_empty_acc": train_non_empty_acc,
            "train_color_obj_acc": train_color_obj_acc,

            "val_loss": val_loss,
            "val_object_loss": val_object_loss,
            "val_color_loss": val_color_loss,
            "val_object_acc": val_object_acc,
            "val_color_acc": val_color_acc,
            "val_non_empty_acc": val_non_empty_acc,
            "val_color_obj_acc": val_color_obj_acc,
            "val_score": val_score
        })

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Score: {val_score:.4f} | "
            f"ObjAcc: {val_object_acc:.4f} | "
            f"NonEmptyAcc: {val_non_empty_acc:.4f} | "
            f"ColorAcc: {val_color_acc:.4f} | "
            f"ColorObjAcc: {val_color_obj_acc:.4f}"
        )

        if val_score > best_score:
            best_score = val_score
            early_stop_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "object2idx": OBJECT2IDX,
                    "color2idx": COLOR2IDX,
                    "best_score": best_score,
                    "epoch": epoch + 1,
                    "config": {
                        "image_size": IMAGE_SIZE,
                        "grid_size": GRID_SIZE,
                        "batch_size": BATCH_SIZE,
                        "lr": LR,
                        "weight_decay": WEIGHT_DECAY
                    }
                },
                model_path
            )

            print("Saved best model")
            print(f"Best Score: {best_score:.4f}")

        else:
            early_stop_counter += 1
            print(f"Early stop: {early_stop_counter}/{PATIENCE}")

            if early_stop_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("Training finished.")
    print("Best score:", best_score)
    print("Model saved to:", model_path)

    upload_to_wandb(
        history=history,
        best_score=best_score
    )

    print("WandB visualization upload finished.")
