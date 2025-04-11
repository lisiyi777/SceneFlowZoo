import argparse
from pathlib import Path
import torch
import numpy as np
from mmengine import Config
import tempfile

from models.whole_batch_optimization.checkpointing.model_loader import OptimCheckpointModelLoader
from dataloaders import TorchFullFrameInputSequence, BaseDataset
import dataloaders
from bucketed_scene_flow_eval.utils import load_json
from bucketed_scene_flow_eval.interfaces import AbstractDataset
import tqdm
import pandas as pd
import pyarrow.feather as feather
import os
import shutil
import cv2


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # delete file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # delete folder
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")


def render_RGB_feature(
    full_sequence: TorchFullFrameInputSequence,
    output_folder: Path,
    global_start_idx: int,
):
    for local_idx in range(len(full_sequence)):
        colors = full_sequence.get_full_pc_rgb(local_idx)
        
        # Calculate the global index for this frame
        global_idx = global_start_idx + local_idx

        colors = colors.detach().cpu().numpy()
        save_df = pd.DataFrame({
            "r": colors[:,0],
            "g": colors[:,1],
            "b": colors[:,2],
        })

        feather_file = f"{str(global_idx).zfill(4)}.feather"

        save_path = output_folder / feather_file

        save_df.to_feather(save_path)


def save_depth_images(
    full_sequence: TorchFullFrameInputSequence,
    output_folder: Path,
    global_start_idx: int,
    camera_views: list[str],
):
    """
    Save depth images for each camera view.
    
    Args:
        full_sequence: The sequence containing point cloud and projection data
        output_folder: Base folder to save depth images
        global_start_idx: The global index of the first frame in this sequence
        camera_views: List of camera view names
    """
    for local_idx in tqdm.tqdm(range(len(full_sequence)), desc="Rendering Depth Images"):
        # Calculate the global index for this frame
        global_idx = global_start_idx + local_idx
        
        # Get the point cloud in ego frame
        ego_pc = full_sequence.get_full_ego_pc(local_idx)
        ego_pc = ego_pc.detach().cpu().numpy()
        
        # Get the projected points and mask for each camera view
        projected_points = full_sequence.rgb_projected_points[local_idx]
        projected_points_mask = full_sequence.rgb_projected_points_mask[local_idx]
        
        # Get the RGB images to determine image dimensions
        rgb_images = full_sequence.rgb_images[local_idx]
        
        # Process each camera view
        for cam_idx, cam_name in enumerate(camera_views):
            # Skip if this camera view doesn't exist
            if cam_idx >= len(projected_points):
                continue
                
            # Get the projected points and mask for this camera
            cam_proj_points = projected_points[cam_idx].detach().cpu().numpy()
            cam_proj_mask = projected_points_mask[cam_idx].detach().cpu().numpy()
            
            # Get the RGB image dimensions for this camera
            if cam_idx < len(rgb_images):
                img_height, img_width = rgb_images[cam_idx].shape[1:3]
            else:
                # Default dimensions if image not available
                img_height, img_width = 1024, 1024
            
            # Create a depth image
            depth_img = np.zeros((img_height, img_width), dtype=np.float32)
            
            # Calculate depth for each projected point
            valid_indices = np.where(cam_proj_mask > 0)[0]
            if len(valid_indices) > 0:
                valid_proj_points = cam_proj_points[valid_indices].astype(np.int32)
                
                valid_3d_points = ego_pc[valid_indices]
                
                camera_pose = (full_sequence.rgb_poses_sensor_to_ego[local_idx][cam_idx]).T
                camera_pose_inv = camera_pose.inverse()
                camera_pose_inv = camera_pose_inv.cpu()
                homog_points = np.concatenate([valid_3d_points, np.ones((len(valid_3d_points), 1))], axis=1)
                camera_points = (camera_pose_inv @ homog_points.T).T
                # Calculate depth as z-coordinate in camera frame
                depths = camera_points[:, 2]
                
                # Filter points that are within image bounds
                in_bounds = (
                    (valid_proj_points[:, 0] >= 0) & 
                    (valid_proj_points[:, 0] < img_width) & 
                    (valid_proj_points[:, 1] >= 0) & 
                    (valid_proj_points[:, 1] < img_height)
                )
                
                valid_proj_points = valid_proj_points[in_bounds]
                depths = depths[in_bounds]
                
                depth_img[valid_proj_points[:, 1], valid_proj_points[:, 0]] = depths
                print(depth_img.max())
                print(depth_img.min())

            cam_dir = output_folder / cam_name
            cam_dir.mkdir(parents=True, exist_ok=True)
            
            depth_file = cam_dir / f"{str(global_idx).zfill(4)}.png"
            cv2.imwrite(str(depth_file), depth_img)


def load_dataset_info(cfg: Config) -> BaseDataset:
    dataset = dataloaders.construct_dataset(cfg.test_dataset.name, cfg.test_dataset.args)
    return dataset


def make_custom_config(
    root_config: Path,
    sequence_id: str,
    sequence_length: int,
) -> Config:
    custom_cfg_content = f"""
_base_="{root_config.absolute()}"
test_dataset=dict(
    args=dict(
        log_subset=["{sequence_id}"],
        subsequence_length={sequence_length},
        use_cache=False,
    )
)
"""
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir)
        custom_cfg = path / f"{root_config.stem}_{sequence_id}.py"
        custom_cfg.write_text(custom_cfg_content)
        return Config.fromfile(custom_cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("sequence_id_to_length", type=Path)
    args = parser.parse_args()

    sequence_lengths = load_json(args.sequence_id_to_length)
    subsequence_length = 5
    
    # Define camera views
    camera_views = [
        "ring_side_left",
        "ring_front_left",
        "ring_front_center",
        "ring_front_right",
        "ring_side_right",
        "ring_rear_left",
        "ring_rear_right",
    ]

    for sequence_id, sequence_length in tqdm.tqdm(sequence_lengths.items(), desc="Processing sequences"):
        print(f"\nProcessing sequence: {sequence_id}")
        
        config = make_custom_config(args.config, sequence_id, subsequence_length)
        
        dataset = load_dataset_info(config)
        
        # Create output directories
        sequence_output_folder = args.output_folder / sequence_id / "sensors"
        lidar_color_folder = sequence_output_folder / "lidar_color"
        depth_image_folder = sequence_output_folder / "depth_image"
        
        lidar_color_folder.mkdir(parents=True, exist_ok=True)
        depth_image_folder.mkdir(parents=True, exist_ok=True)
        data_length = len(dataset)
        
        for i in tqdm.tqdm(range(0, data_length, subsequence_length), desc="Rendering LiDAR Colors"):
            subsequence = dataset[i].to("cuda")
            render_RGB_feature(subsequence, lidar_color_folder, i)
            # save_depth_images(subsequence, depth_image_folder, i, camera_views)

        subsequence = dataset[data_length-1].to("cuda")
        render_RGB_feature(subsequence, lidar_color_folder, data_length-1)

if __name__ == "__main__":
    main()
