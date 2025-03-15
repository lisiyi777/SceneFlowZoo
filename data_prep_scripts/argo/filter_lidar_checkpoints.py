import os
import json
from pathlib import Path

def generate_scene_log_json(checkpoints_dir, output_json):
    scene_dict = {}
    
    for folder in Path(checkpoints_dir).iterdir():
        if folder.is_dir() and folder.name.startswith("job_"):
            scene_log = folder.name.replace("job_", "").replace("seqlen160idx000000", "").strip()
            checkpoint_file = folder / "best_weights.pth"
            
            # Check if the checkpoint file exists and is not empty
            if checkpoint_file.exists() and checkpoint_file.stat().st_size > 0:
                scene_dict[scene_log] = 160  # Fixed sequence length
    
    with open(output_json, "w") as f:
        json.dump(scene_dict, f, indent=4)

    print(f"JSON file saved at {output_json}")

checkpoints_directory = "/efs/000106_gigachad_lidar_train"
output_json_file = "./euler_sequence_length.json"

generate_scene_log_json(checkpoints_directory, output_json_file)
