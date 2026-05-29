import os
import json

save_dir = # r""
map_num = 400

def generate_instructions(meta):
    cube_color_name = meta["cube"]["color_name"]
    cube_pos = meta["cube"]["position"]
    key_color_name = meta["key"]["color_name"]
    key_pos = meta["key"]["position"]
    exit_pos = meta["exit"]

    instructions = []

    target_info = {
        "cube": {
            "color_name": cube_color_name,
            "target_pos": cube_pos
        },
        "key": {
            "color_name": key_color_name,
            "target_pos": key_pos
        },
        "exit": {
            "color_name": None,
            "target_pos": exit_pos
        }
    }

    template_map = {
        "cube": [
            "go to the {target}",
            "move to the {target}",
        ],
        "key": [
            "go to the {target}",
            "move to the {target}",
        ],
        "exit": [
            "go to the {target}",
            "move to the {target}"
        ]
    }

    for target_type, info in target_info.items():
        color_name = info["color_name"]
        target_pos = info["target_pos"]

        target_texts = [target_type]

        if color_name is not None:
            target_texts.append(f"{color_name} {target_type}")

        for target_text in target_texts:
            for template in template_map[target_type]:
                instructions.append({
                    "instruction": template.format(target=target_text),
                    "type": "single",
                    "targets": [
                        {"target_type": target_type, "target_pos": target_pos}
                    ]
                })

    return instructions

all_map_instructions = []

for map_id in range(1, map_num + 1):
    json_name = f"map_{map_id:02d}.json"
    json_path = os.path.join(save_dir, json_name)

    if not os.path.exists(json_path):
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    instructions = generate_instructions(meta)

    all_map_instructions.append({
        "image_name": meta["image_name"],
        "instructions": instructions
    })

output_path = os.path.join(save_dir, "all_instructions.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_map_instructions, f, indent=2)

print("total number of instructions:", len(all_map_instructions))
