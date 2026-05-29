import os, pickle
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from A_Star_Algorithm import get_step, build_grid_from_meta, A_Star_trajectory

train_maps_save_dir = r"C:\Users\Admin\Desktop\File\OENG1088\VLA\final_model\train_maps"
validate_maps_save_dir = r"C:\Users\Admin\Desktop\File\OENG1088\VLA\final_model\validate_maps"
mapcnn_model_path = r"C:\Users\Admin\Desktop\File\OENG1088\VLA\final_model\final_mapcnn_model\final_mapcnn_model.pth"
heatmap_model_path = r"C:\Users\Admin\Desktop\File\OENG1088\VLA\final_model\final_heatmap_model\final_heatmap_model.pth"
pca_path = r"C:\Users\Admin\Desktop\File\OENG1088\VLA\final_model\text_joint_pca_512.pkl"
action_dataset_save_dir = r"C:\Users\Admin\Desktop\File\OENG1088\VLA\final_model\action_train_dataset"
os.makedirs(action_dataset_save_dir, exist_ok=True)

ACTIONS = [
    "right", "right_up", "up", "left_up",
    "left", "left_down", "down", "right_down"
]
action2idx = {a: i for i, a in enumerate(ACTIONS)}

class MapCNN(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        feat = self.conv(x) 
        return feat

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

def load_map_image(image_path, image_size=80):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img

def text_embedding(text):
    with torch.no_grad ():
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        outputs = bert_model(**inputs)
        text_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
    return text_embedding

COLOR_MAPS = ["red", "blue", "green", "yellow", "cyan"]
def extract_color(instruction_text):
    text = instruction_text.lower()
    for c in COLOR_MAPS:
        if c in text:
            return c
    return None

def build_text_joint_feature(instruction_text, target_name):
    with torch.no_grad():
        instr_feature = text_embedding(instruction_text).numpy().astype(np.float32)
        color_name = extract_color(instruction_text)
        if color_name is None:
            color_feat = np.zeros((768,), dtype=np.float32)
        else:
            color_feat = text_embedding(color_name).numpy().astype(np.float32)
        target_feat = text_embedding(target_name).numpy().astype(np.float32)
    text_joint = np.concatenate(
        [instr_feature, color_feat, target_feat],
        axis=0
    ).reshape(1, -1)
    text_pca = pca.transform(text_joint).astype(np.float32)
    text_tensor = torch.tensor(text_pca, dtype=torch.float32)
    return text_tensor

def predict_heatmap(image_path, instruction_text, target_name):
    with torch.no_grad():
        img_tensor = load_map_image(image_path)
        map_feat = mapcnn_model(img_tensor)
        text_feat = build_text_joint_feature(instruction_text, target_name)
        pred_heatmap = heatmap_model(map_feat, text_feat)
        pred_heatmap = pred_heatmap.squeeze(0).numpy().astype(np.float32)
    return pred_heatmap

def process_split(root_dir, X_heatmap_pred, X_pos, Y_action):
    instruction_path = os.path.join(root_dir, "all_instructions.json")

    with open(instruction_path, "r", encoding="utf-8") as f:
        all_map_instructions = json.load(f)

    for map_item in all_map_instructions:
        image_name = map_item["image_name"]
        image_path = os.path.join(root_dir, image_name)

        meta_path = os.path.join(root_dir, image_name.replace(".png", ".json"))
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        start = tuple(meta["start"])
        grid = build_grid_from_meta(meta, rows=10, cols=10)

        for inst_item in map_item["instructions"]:
            instruction_text = inst_item["instruction"]
            targets = inst_item["targets"]

            target = targets[0]
            target_pos = tuple(target["target_pos"])
            target_name = target["target_type"]

            pred_heatmap = predict_heatmap(
                image_path,
                instruction_text,
                target_name
            )

            path = A_Star_trajectory(grid, start, target_pos)
            steps = get_step(path)

            for i in range(len(steps)):
                current_pos = path[i]
                next_action = steps[i]
                pos_feature = np.array(
                    [current_pos[0] / 9.0, current_pos[1] / 9.0],
                    dtype=np.float32
                )
                action_label = action2idx[next_action]
                X_heatmap_pred.append(pred_heatmap)
                X_pos.append(pos_feature)
                Y_action.append(action_label)

mapcnn_ckpt = torch.load(mapcnn_model_path)
state_dict = mapcnn_ckpt["model_state_dict"]
conv_state_dict = {k: v for k, v in state_dict.items() if k.startswith("conv.")}
mapcnn_model = MapCNN()
mapcnn_model.load_state_dict(conv_state_dict)
mapcnn_model.eval()

heatmap_model = HeatmapHead()
heatmap_ckpt = torch.load(heatmap_model_path)
heatmap_model.load_state_dict(heatmap_ckpt["model_state_dict"])
heatmap_model.eval()

bert_model = BertModel.from_pretrained("bert-base-cased")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
bert_model.eval()

with open(pca_path, "rb") as f:
    pca = pickle.load(f)

X_train_heatmap_pred = []
X_train_pos = []
Y_train_action = []

X_validate_heatmap_pred = []
X_validate_pos = []
Y_validate_action = []

process_split(
    train_maps_save_dir,
    X_train_heatmap_pred,
    X_train_pos,
    Y_train_action
)

process_split(
    validate_maps_save_dir,
    X_validate_heatmap_pred,
    X_validate_pos,
    Y_validate_action
)

X_train_heatmap_pred = np.array(X_train_heatmap_pred, dtype=np.float32)
X_train_pos = np.array(X_train_pos, dtype=np.float32)
Y_train_action = np.array(Y_train_action, dtype=np.int64)

X_validate_heatmap_pred = np.array(X_validate_heatmap_pred, dtype=np.float32)
X_validate_pos = np.array(X_validate_pos, dtype=np.float32)
Y_validate_action = np.array(Y_validate_action, dtype=np.int64)

np.save(os.path.join(action_dataset_save_dir, "X_train_action_heatmap_pred.npy"), X_train_heatmap_pred)
np.save(os.path.join(action_dataset_save_dir, "X_train_action_pos.npy"), X_train_pos)
np.save(os.path.join(action_dataset_save_dir, "Y_train_action.npy"), Y_train_action)

np.save(os.path.join(action_dataset_save_dir, "X_validate_action_heatmap_pred.npy"), X_validate_heatmap_pred)
np.save(os.path.join(action_dataset_save_dir, "X_validate_action_pos.npy"), X_validate_pos)
np.save(os.path.join(action_dataset_save_dir, "Y_validate_action.npy"), Y_validate_action)

print("Second-stage action dataset saved successfully.")
print("X_train_action_heatmap:", X_train_heatmap_pred.shape)
print("X_train_action_pos:", X_train_pos.shape)
print("Y_train_action:", Y_train_action.shape)
print("X_validate_action_heatmap:", X_validate_heatmap_pred.shape)
print("X_validate_action_pos:", X_validate_pos.shape)
print("Y_validate_action:", Y_validate_action.shape)