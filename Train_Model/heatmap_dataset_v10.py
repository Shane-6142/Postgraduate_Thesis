import torch
import torch, os, json, cv2, re, pickle
import numpy as np
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA

train_maps_save_dir = # r""
validate_maps_save_dir = # r""
mapcnn_model_dir = # "final_mapcnn_model.pth"
save_dir = # r""
os.makedirs(save_dir, exist_ok=True)

pca = PCA(n_components=512, random_state=0)

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
    
def load_map_image(image_path, image_size=80):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img

def make_gaussian_heatmap(target_pos, sigma=1.0):
    tx, ty = target_pos
    heatmap = np.zeros((10, 10))

    for y in range(10):
        for x in range(10):
            heatmap[y, x] = np.exp(-((x - tx) ** 2 + (y - ty) ** 2) / (2 * sigma ** 2))

    heatmap /= heatmap.max()
    return heatmap

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

def process_split(root_dir, X_map_feat, X_text, Y_heatmap):
    instruction_path = os.path.join(root_dir, "all_instructions.json")
    with open(instruction_path, "r", encoding="utf-8") as f:
        all_map_instructions = json.load(f)
    for map_item in all_map_instructions:
        image_name = map_item["image_name"]
        image_path = os.path.join(root_dir, image_name)
        with torch.no_grad():
            img_tensor = load_map_image(image_path)
            map_feature = mapcnn_model(img_tensor).squeeze(0)
        for inst_item in map_item["instructions"]:
            instruction_text = inst_item["instruction"]
            targets = inst_item["targets"]
            text_feature = text_embedding(instruction_text)
            color_name = extract_color(instruction_text)
            if color_name is None:
                color_feat = np.zeros((768,), dtype=np.float32)
            else:
                color_feat = text_embedding(color_name)
            target = targets[0]
            target_name = target["target_type"]
            target_feat = text_embedding(target_name)
            text_feature_con = np.concatenate(
                [text_feature, color_feat, target_feat],
                axis=0
            )
            target_pos = tuple(target["target_pos"])
            heatmap = make_gaussian_heatmap(target_pos)
            X_map_feat.append(map_feature)
            X_text.append(text_feature_con)
            Y_heatmap.append(heatmap)

ckpt = torch.load(mapcnn_model_dir)
state_dict = ckpt["model_state_dict"]
conv_state_dict = {k: v for k, v in state_dict.items() if k.startswith("conv.")}
mapcnn_model = MapCNN()
mapcnn_model.load_state_dict(conv_state_dict)
mapcnn_model.eval()

bert_model = BertModel.from_pretrained("bert-base-cased")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
bert_model.eval()

X_train_map_feat = []
X_train_text = []
Y_train_heatmap = []

X_validate_map_feat = []
X_validate_text = []
Y_validate_heatmap = []

process_split(
    train_maps_save_dir,
    X_train_map_feat,
    X_train_text,
    Y_train_heatmap
)

process_split(
    validate_maps_save_dir,
    X_validate_map_feat,
    X_validate_text,
    Y_validate_heatmap
)

X_train_map_feat = np.array(X_train_map_feat, dtype=np.float32)
X_train_text = np.array(X_train_text, dtype=np.float32)
Y_train_heatmap = np.array(Y_train_heatmap, dtype=np.float32)

X_validate_map_feat = np.array(X_validate_map_feat, dtype=np.float32)
X_validate_text = np.array(X_validate_text, dtype=np.float32)
Y_validate_heatmap = np.array(Y_validate_heatmap, dtype=np.float32)

X_train_text = pca.fit_transform(X_train_text).astype(np.float32)
X_validate_text = pca.transform(X_validate_text).astype(np.float32)

with open("text_joint_pca_512.pkl", "wb") as f:
    pickle.dump(pca, f)

np.save(os.path.join(save_dir, "X_train_map_feat.npy"), X_train_map_feat)
np.save(os.path.join(save_dir, "X_train_text.npy"), X_train_text)
np.save(os.path.join(save_dir, "Y_train_heatmap.npy"), Y_train_heatmap)

np.save(os.path.join(save_dir, "X_validate_map_feat.npy"), X_validate_map_feat)
np.save(os.path.join(save_dir, "X_validate_text.npy"), X_validate_text)
np.save(os.path.join(save_dir, "Y_validate_heatmap.npy"), Y_validate_heatmap)

print("New heatmap dataset saved successfully.")
print("X_train_map_feat:", X_train_map_feat.shape)
print("X_train_text:", X_train_text.shape)
print("Y_train_heatmap:", Y_train_heatmap.shape)

print("X_validate_map_feat:", X_validate_map_feat.shape)
print("X_validate_text:", X_validate_text.shape)
print("Y_validate_heatmap:", Y_validate_heatmap.shape)
