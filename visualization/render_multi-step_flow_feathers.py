import argparse
from pathlib import Path
import torch
import numpy as np

from models.whole_batch_optimization.checkpointing.model_loader import OptimCheckpointModelLoader
from dataloaders import TorchFullFrameInputSequence
from bucketed_scene_flow_eval.datastructures import (
    O3DVisualizer,
    PointCloud,
    TimeSyncedSceneFlowFrame,
    SupervisedPointCloudFrame,
    ColoredSupervisedPointCloudFrame,
)
from bucketed_scene_flow_eval.interfaces import AbstractDataset
from visualization.vis_lib import BaseCallbackVisualizer
from bucketed_scene_flow_eval.utils import load_json, save_json
from dataclasses import dataclass
from models.mini_batch_optimization import EulerFlowModel
from models.components.neural_reps import ModelFlowResult, ModelOccFlowResult, QueryDirection
import open3d as o3d
import json
import tqdm
import multiprocessing as mp
import pandas as pd
import pyarrow.feather as feather
from render_flow_feathers import SceneFlowData 

from sklearn.neighbors import NearestNeighbors

def save_multi_step_flow_to_feather(save_path: Path, flows: dict[int, np.ndarray], mask: np.ndarray):
    full_flows = {
        step: np.zeros((mask.shape[0], 3), dtype=np.float32) for step in flows.keys()
    }
    for step, flow in flows.items():
        full_flows[step][mask] = flow

    output_dict = {
        "is_valid": mask.astype(bool),
        "classes_0": np.full(mask.shape[0], -1, dtype=np.int8),
        # Save step 1 flow in traditional way without step info
        "flow_tx_m": full_flows[1][:, 0],
        "flow_ty_m": full_flows[1][:, 1],
        "flow_tz_m": full_flows[1][:, 2],
    }
    
    # Save flows for steps > 1 with step information
    for step, flow in full_flows.items():
        if step > 1:
            output_dict.update({
                f"flow_tx_m_step{step}": flow[:, 0],
                f"flow_ty_m_step{step}": flow[:, 1], 
                f"flow_tz_m_step{step}": flow[:, 2],
            })

    output_df = pd.DataFrame(output_dict)
    feather.write_feather(output_df, save_path)


@dataclass
class MultiStepSceneFlowData:
    points: np.ndarray
    colors: np.ndarray
    flows: list[np.ndarray]  
    mask: np.ndarray
    timestamp: str

    def __post_init__(self):
        assert (
            self.points.shape[0] == self.colors.shape[0]
        ), f"{self.points.shape} != {self.colors.shape}"
        
        for step, flow in enumerate(self.flows, start=1):
            assert (
                self.points.shape[0] == flow.shape[0]
            ), f"{self.points.shape} != {flow.shape} for step {step}"
            assert (
                np.sum(self.mask) == flow.shape[0]
            ), f"Number of valid points in mask ({np.sum(self.mask)}) does not match flow array size ({flow.shape[0]}) for step {step}"

    def save(self, parent_folder: Path, idx: int):
        parent_folder.mkdir(parents=True, exist_ok=True)
        feather_path = parent_folder / f"{self.timestamp}.feather"
        
        all_flows = {step: flow for step, flow in enumerate(self.flows, start=1)}
        save_multi_step_flow_to_feather(feather_path, all_flows, self.mask)

def save_result(result: MultiStepSceneFlowData, parent_folder: Path, idx: int):
    result.save(parent_folder, idx)

def knn_match_ratio(warped_pc: np.ndarray, target_pc: np.ndarray, threshold=0.3):
    knn = NearestNeighbors(n_neighbors=1).fit(target_pc)
    dists, _ = knn.kneighbors(warped_pc)
    match_ratio = (dists < threshold).sum() / len(dists)
    return match_ratio, dists

def render_multi_step_flows(
    model: EulerFlowModel,
    full_sequence: TorchFullFrameInputSequence,
    base_dataset: AbstractDataset, 
    output_folder: Path,
    rollout_steps: int = 5
) -> list[MultiStepSceneFlowData]:
    base_dataset_full_sequence = base_dataset[full_sequence.sequence_idx]
    results: list[MultiStepSceneFlowData] = []

    model.model = model.model.eval()
    with torch.no_grad():
        for idx in tqdm.tqdm(range(len(base_dataset_full_sequence) - 1), desc="Rendering Multi-step Flows"):
            torch_query_points = full_sequence.get_global_pc(idx)      # (PadN, 3)
            torch_full_mask = full_sequence.get_full_pc_mask(idx)      # (PadN,)
            scene_flow_frame = base_dataset_full_sequence[idx]

            multi_step_flows = []
            current_points = torch_query_points.clone()

            ego_to_global = full_sequence.get_pc_poses_ego_to_global(idx)
            global_to_ego = torch.inverse(ego_to_global[:3, :3])

            for step in range(min(rollout_steps, len(base_dataset_full_sequence) - idx - 1)):
                query_result: ModelFlowResult = model.model(
                    current_points, idx + step, len(full_sequence), QueryDirection.FORWARD,
                )
                
                current_points = current_points + query_result.flow

                accumulated_global_flows = current_points - torch_query_points
                flow_ego = (global_to_ego @ accumulated_global_flows.T).T
                multi_step_flows.append(flow_ego.detach().cpu().numpy())

                # current_points_np = current_points.detach().cpu().numpy()
                # if step == 0:
                #     glb_pc = full_sequence.get_global_pc_gt_flowed(idx)
                # else:
                #     ego_pc = full_sequence.get_ego_pc_gt_multi_step_flowed(idx, step-1)
                #     glb_pc = (ego_to_global[:3, :3] @ ego_pc.T).T + ego_to_global[:3, 3]  # Transform to global

                # torch_current_points_np = glb_pc.detach().cpu().numpy()

                # if idx + step < len(base_dataset_full_sequence):
                #     target_pc_np = base_dataset_full_sequence[idx + step + 1].pc.global_pc.points
                #     # if step == 0:
                #     #     match_ratio, dists = knn_match_ratio(torch_query_points.detach().cpu().numpy(), target_pc_np)
                #     #     print(f"idx={idx} | baseline match ratio = {match_ratio:.3f}")
                #     match_ratio, dists = knn_match_ratio(torch_current_points_np, current_points_np)
                #     print(f"idx={idx} | step={step} | match ratio = {match_ratio:.3f}")

            # Convert to numpy arrays
            mask_np = torch_full_mask.detach().cpu().numpy()
            pc_frame: SupervisedPointCloudFrame = scene_flow_frame.pc
            if isinstance(pc_frame, ColoredSupervisedPointCloudFrame):
                color_np = pc_frame.colors[pc_frame.mask]
            else:
                color_np = np.ones_like(multi_step_flows[0])
            pc_np = pc_frame.global_pc.points

            # Save result
            results.append(
                MultiStepSceneFlowData(
                    points=pc_np,
                    colors=color_np,
                    flows=multi_step_flows,
                    mask=mask_np,
                    timestamp=f"{scene_flow_frame.log_timestamp}"
                )
            )

    print("Saving results")
    arguments_lst = [(result, output_folder, idx) for idx, result in enumerate(results)]
    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(save_result, arguments_lst)
    return results



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("sequence_id_to_length",type=Path)
    args = parser.parse_args()

    sequence_lengths = load_json(args.sequence_id_to_length)

    for sequence_id, sequence_length in tqdm.tqdm(sequence_lengths.items(), desc="Processing sequences"):
        print(f"\nProcessing sequence: {sequence_id}")
        
        model_loader = OptimCheckpointModelLoader.from_checkpoint_dirs(
            args.config, args.checkpoint_root, sequence_id, args.sequence_id_to_length
        )
        
        model, full_sequence, base_dataset = model_loader.load_model()
        model: EulerFlowModel
        
        sequence_output_folder = args.output_folder / sequence_id
        sequence_output_folder.mkdir(parents=True, exist_ok=True)
        
        render_multi_step_flows(model, full_sequence, base_dataset, sequence_output_folder)
            


if __name__ == "__main__":
    main()
