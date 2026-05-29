import os
import numpy as np
import matplotlib.pyplot as plt
import random
import json
from collections import deque
from draw_key import draw_key

# save_dir = r"C:\Users\Admin\Desktop\File\OENG1088\VLA\final_model\train_maps"
save_dir = r"C:\Users\Admin\Desktop\File\OENG1088\VLA\final_model\validate_maps"

map_num = 400

COLOR_MAP = {
    "empty": (255, 255, 255),
    "obstacles": (0, 0, 0),
    "start": (255, 165, 0),
    "exit": (255, 0, 255)
}

color_rgb_map = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255)
}

def generate_random_color(exclude_colors = []):
    color_list = list(color_rgb_map.keys())
    while True:
        random_color_name = random.choice(color_list)
        color = color_rgb_map[random_color_name]
        if color not in exclude_colors:
            return random_color_name, color
        
def protect_neighbor(pos):
    x, y = pos
    neighbors = set()
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < cols and 0 <= ny < rows:
            neighbors.add((nx, ny))
    return neighbors

map_size=(10,10)
rows, cols = map_size

wall_positions = []
for x in range(cols):
    wall_positions.append((x, 0))
    wall_positions.append((x, rows - 1))
for y in range(1, rows - 1):
    wall_positions.append((0, y))
    wall_positions.append((cols - 1, y))

def generate_empty_map():
    empty_map = np.full((rows, cols, 3), COLOR_MAP["empty"], dtype=np.uint8)
    return empty_map

def is_walkable(img, x, y):
    return tuple(img[y, x]) != COLOR_MAP["obstacles"]

def is_map_reachable(img, start_pos, cube_pos, key_pos, exit_pos):
    queue = deque([start_pos])
    visited = set([start_pos])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            if (nx, ny) in visited:
                continue
            if not is_walkable(img, nx, ny):
                continue
            visited.add((nx, ny))
            queue.append((nx, ny))
    targets = [cube_pos, key_pos, exit_pos]
    return all(target in visited for target in targets)

for map_id in range(map_num):
    while True:
        empty_img = generate_empty_map()
        start_pos = random.choice(wall_positions)
        while True:
            exit_pos = random.choice(wall_positions)
            if exit_pos != start_pos:
                break

        empty_img[start_pos[1], start_pos[0]] = COLOR_MAP["start"]
        empty_img[exit_pos[1], exit_pos[0]] = COLOR_MAP["exit"]
        occupied_positions = {start_pos, exit_pos}
        occupied_positions.update(protect_neighbor(start_pos))
        occupied_positions.update(protect_neighbor(exit_pos))
        
        obstacle_positions = []
        while len(obstacle_positions) < 10:
            x = random.randint(0, cols - 1)
            y = random.randint(0, rows - 1)

            if (x, y) in occupied_positions:
                continue

            obstacle_positions.append((x, y))
            occupied_positions.add((x, y))
            empty_img[y, x] = COLOR_MAP["obstacles"]
        
        cube_color_name, cube_color = generate_random_color()

        while True:
            cube_x = random.randint(0, cols - 1)
            cube_y = random.randint(0, rows - 1)
            if (cube_x, cube_y) not in occupied_positions:
                break
        
        empty_img[cube_y, cube_x] = cube_color
        occupied_positions.add((cube_x, cube_y))
        cube_pos = (cube_x, cube_y)

        key_color_name, key_color = generate_random_color()

        while True:
            key_x = random.randint(0, cols - 1)
            key_y = random.randint(0, rows - 1)
            if (key_x, key_y) not in occupied_positions:
                break

        occupied_positions.add((key_x, key_y))
        key_pos = (key_x, key_y)

        if is_map_reachable(empty_img, start_pos, cube_pos, key_pos, exit_pos):
            break

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(empty_img, extent=[0, cols, rows, 0])
    scale_x = cols / 10
    scale_y = rows / 10
    key_col_norm = tuple(c / 255.0 for c in key_color)
    key = draw_key(ax, key_x, key_y, scale_x, scale_y, key_col_norm, zorder=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(which='both', color='gray', linestyle='--', linewidth=1)

    img_name = f"map_{map_id + 1:02d}.png"
    img_path = os.path.join(save_dir, img_name)
    plt.savefig(img_path, bbox_inches='tight', dpi=100)
    plt.close(fig)

    meta = {
        "map_id": map_id + 1,
        "image_name": img_name,
        "start": [start_pos[0], start_pos[1]],
        "exit": [exit_pos[0], exit_pos[1]],
        "cube": {
            "position": [cube_x, cube_y],
            "color_name": cube_color_name
        },
        "key": {
            "position": [key_x, key_y],
            "color_name": key_color_name
        },
        "obstacles": [[x, y] for (x, y) in obstacle_positions]
    }

    json_name = f"map_{map_id + 1:02d}.json"
    json_path = os.path.join(save_dir, json_name)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
