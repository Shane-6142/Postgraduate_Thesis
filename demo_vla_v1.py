import torch, os, json, cv2, pickle, re
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog
from transformers import BertTokenizer, BertModel
from draw_key import draw_key
from agent_class_demo_version import Agent

test_maps_save_dir = r"C:\Users\Admin\Desktop\File\OENG1088\VLA\Demo\Test_maps"
mapcnn_model_path = r"C:\Users\Admin\Desktop\File\OENG1088\VLA\final_model\final_mapcnn_model\final_mapcnn_model.pth"
heatmap_model_path = r"C:\Users\Admin\Desktop\File\OENG1088\VLA\final_model\final_heatmap_model\final_heatmap_model.pth"
action_model_path = r"C:\Users\Admin\Desktop\File\OENG1088\VLA\final_model\final_action_model\final_action_model.pth"
pca_path = r"C:\Users\Admin\Desktop\File\OENG1088\VLA\final_model\text_joint_pca_512.pkl"
instruction_path = os.path.join(test_maps_save_dir, "all_instructions.json")

MAX_STEPS = 20

ACTIONS = [
    "right", "right_up", "up", "left_up",
    "left", "left_down", "down", "right_down"
]
action2idx = {a: i for i, a in enumerate(ACTIONS)}
idx2action = {i: a for a, i in action2idx.items()}

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
        return self.conv(x)
    
def load_map_image(image_path, image_size=80):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img

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
    
class ActionHead(nn.Module):
    def __init__(self, heatmap_dim=100, pos_dim=2, hidden_dim1=128, hidden_dim2=64, num_classes=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(heatmap_dim + pos_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, num_classes)
        )

    def forward(self, heatmap, pos_feat):
        heatmap_flat = heatmap.view(heatmap.size(0), -1)
        x = torch.cat([heatmap_flat, pos_feat], dim=-1)
        return self.net(x)

def find_map_files(root_dir, map_number):
    img_name = f"map_{map_number:02d}.png"
    json_name = f"map_{map_number:02d}.json"
    img_path = os.path.join(root_dir, img_name)
    json_path = os.path.join(root_dir, json_name)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Cannot find image file: {img_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cannot find json file: {json_path}")
    return img_path, json_path

def text_embedding(text):
    with torch.no_grad ():
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        outputs = bert_model(**inputs)
        text_feature = outputs.last_hidden_state[:, 0, :].squeeze(0)
    return text_feature
    
def load_map_image(image_path, image_size=80):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img

def input_map_number():
    root = tk.Tk()
    root.withdraw()
    map_number = simpledialog.askinteger(
        "Select Map",
        "Enter map number from 1 to 20:",
        minvalue=1,
        maxvalue=20
    )
    root.destroy()
    if map_number is None:
        raise RuntimeError("User cancelled map selection.")
    return map_number

def input_instruction():
    root = tk.Tk()
    root.withdraw()
    raw_instruction = simpledialog.askstring(
        "Custom Instruction",
        "Enter your instruction:"
    )
    root.destroy()
    if not raw_instruction or raw_instruction.strip() == "":
        raise RuntimeError("No instruction entered.")
    return raw_instruction.strip()

def is_compound_instruction(instruction_text):
    return re.search(r"\b(and|then)\b", instruction_text.lower()) is not None

def need_pick(instruction_text):
    text = instruction_text.lower()
    return re.search(r"\b(pick|get)\b", text) is not None

def split_instruction(instruction_text):
    parts = re.split(
        r"\b(?:and|then)\b",
        instruction_text,
        flags=re.IGNORECASE
    )
    instructions_list = [p.strip() for p in parts if p.strip() != ""]
    return instructions_list

def infer_target_from_instruction(instruction_text):
    text = instruction_text.lower()
    if "key" in text:
        return "key", key_x, key_y
    if "cube" in text:
        return "cube", cube_x, cube_y
    if "exit" in text:
        return "exit", exit_x, exit_y
    raise ValueError(f"Unknown instruction segment: {instruction_text}")

color_rgb_map = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255)
}
def extract_color(instruction_text):
    text = instruction_text.lower()
    for c in color_rgb_map:
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
    ).reshape(1, -1).astype(np.float32)
    text_pca = pca.transform(text_joint).astype(np.float32)
    text_tensor = torch.tensor(text_pca, dtype=torch.float32)
    return text_tensor

mapcnn_ckpt = torch.load(mapcnn_model_path)
state_dict = mapcnn_ckpt["model_state_dict"]
conv_state_dict = {k: v for k, v in state_dict.items() if k.startswith("conv.")}
mapcnn_model = MapCNN()
mapcnn_model.load_state_dict(conv_state_dict)
mapcnn_model.eval()

bert_model = BertModel.from_pretrained("bert-base-cased")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
bert_model.eval()

heatmap_model = HeatmapHead()
heatmap_ckpt = torch.load(heatmap_model_path)
heatmap_model.load_state_dict(heatmap_ckpt["model_state_dict"])
heatmap_model.eval()

action_model = ActionHead()
action_ckpt = torch.load(action_model_path)
action_model.load_state_dict(action_ckpt["model_state_dict"])
action_model.eval()

with open(pca_path, "rb") as f:
    pca = pickle.load(f)

map_num = input_map_number()
map_path, json_path = find_map_files(test_maps_save_dir, map_num)
print(f"Selected map: {map_num:02d}")

with open(json_path, "r", encoding="utf-8") as f:
    test_map_meta = json.load(f)

start_x, start_y = test_map_meta["start"]
exit_x, exit_y = test_map_meta["exit"]
key_x, key_y = test_map_meta["key"]["position"]
cube_x, cube_y = test_map_meta["cube"]["position"]
key_color_name = test_map_meta["key"]["color_name"]
key_rgb = color_rgb_map[key_color_name]
cube_color_name = test_map_meta["cube"]["color_name"]
cube_rgb = color_rgb_map[cube_color_name]
key_rgb_normalized = [x / 255.0 for x in key_rgb]

map_size=(10,10)
def load_map(map_path):
    img = cv2.imread(map_path)
    img_resized = cv2.resize(img, map_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    return img_rgb
map_img = load_map(map_path)
rows, cols = map_img.shape[:2]
map_img[key_y, key_x] = [255, 255, 255]

plt.ion()
fig, ax = plt.subplots(figsize=(10, 8))
ax_img = ax.imshow(map_img, extent=[0, cols, rows, 0], zorder=0)
scale_x = cols / 10
scale_y = rows / 10
key = draw_key(ax, key_x, key_y, scale_x, scale_y, key_rgb_normalized, zorder=5)
ax.set_xticks(np.arange(0, cols+1, 1))
ax.set_yticks(np.arange(0, rows+1, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(which='both', color='gray', linestyle='--', linewidth=1)
agent = Agent(grid_x=start_x, grid_y=start_y)
ax.add_patch(agent.triangle)
plt.show()
plt.pause(0.5)

raw_instruction = input_instruction()
print(f"Instruction: {raw_instruction}")

if is_compound_instruction(raw_instruction):
    instructions_list = split_instruction(raw_instruction)
else:
    instructions_list = [raw_instruction]

total_steps_all = 0
all_actions = []
success_all = 1

for seg_idx, current_instruction in enumerate(instructions_list):
    print("Current instruction:", current_instruction)

    target_name, target_x, target_y = infer_target_from_instruction(current_instruction)

    need_pickup = need_pick(current_instruction) and target_name in ["key", "cube"]

    with torch.no_grad():
        img_tensor = load_map_image(map_path)
        map_feat = mapcnn_model(img_tensor)
        instr_feature = build_text_joint_feature(
            current_instruction,
            target_name
        )
        pred_heatmap = heatmap_model(map_feat, instr_feature)
        pred_heatmap_np = pred_heatmap.squeeze(0).detach().numpy()

    plt.figure(figsize=(4, 4))
    plt.title(f"Pred Heatmap: {current_instruction}")
    plt.imshow(pred_heatmap_np, cmap="hot")
    plt.colorbar()
    plt.show()

    action_list = []
    segment_steps = 0
    segment_success = 0

    for i in range(MAX_STEPS):
        x, y = agent.get_xy()

        if x == target_x and y == target_y:
            print("Reach the target!")
            segment_success = 1
            break

        pos_feature = torch.tensor(
            [[x / 9.0, y / 9.0]],
            dtype=torch.float32
        )

        with torch.no_grad():
            output = action_model(pred_heatmap, pos_feature).squeeze(0)
            ranked_idx = torch.argsort(output, descending=True).tolist()
            ranked_actions = [idx2action[i] for i in ranked_idx]

        segment_steps += 1
        total_steps_all += 1
        chosen_action = None

        for cand_action in ranked_actions:
            old_x, old_y = agent.get_xy()
            agent.rotate_to(cand_action, ax)
            plt.pause(0.2)
            agent.move(ax, map_img)
            plt.pause(0.2)
            new_x, new_y = agent.get_xy()
            if (new_x != old_x) or (new_y != old_y):
                chosen_action = cand_action
                break
        action_list.append(chosen_action)
        all_actions.append(chosen_action)

        print(
            f"Step {i + 1} | "
            f"location: ({x},{y}) | "
            f"action: {chosen_action} | "
            f"target=({target_x},{target_y})"
        )

        ax.set_title(
            f"Segment {seg_idx + 1} | "
            f"Instruction: {current_instruction} | "
            f"Step {i + 1}"
        )
        plt.pause(0.2)

        new_x, new_y = agent.get_xy()
        if new_x == target_x and new_y == target_y:
            print("Reach the target!")
            segment_success = 1
            break

    if segment_success == 1 and need_pickup:
        if target_name == "key":
            hold_color_rgb = key_rgb
        elif target_name == "cube":
            hold_color_rgb = cube_rgb
        else:
            hold_color_rgb = (255, 0, 255)

        picked = agent.pick(ax, map_img, ax_img, target_name, hold_color_rgb)

        if picked:
            all_actions.append("pickup")
            plt.pause(0.5)

    if segment_success == 0:
        print(f"Segment failed: {current_instruction}")
        success_all = 0
        break

collision_times = agent.collision_times

print("\n================ Demo Summary ================")
print(f"Map: map_{map_num:02d}")
print(f"Instruction: {raw_instruction}")
print(f"Collision Times: {collision_times}")
print(f"Total Steps: {total_steps_all}")
print(f"Actions: {all_actions}")

if success_all == 1:
    print("Instruction completed.")
else:
    print("Task failed.")

plt.ioff()
plt.show()