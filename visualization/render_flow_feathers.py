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


def save_flow_to_feather(save_path: Path, flows: np.ndarray, mask: np.ndarray):
    full_flow = np.zeros((mask.shape[0], 3), dtype=np.float32)
    full_flow[mask] = flows
    output_df = pd.DataFrame(
        {
            "is_valid": mask.astype(bool),
            "flow_tx_m": full_flow[:, 0],
            "flow_ty_m": full_flow[:, 1],
            "flow_tz_m": full_flow[:, 2],
        }
    )
    feather.write_feather(output_df, save_path)

@dataclass
class SceneFlowData:
    points: np.ndarray
    colors: np.ndarray
    flows: np.ndarray
    mask: np.ndarray
    timestamp: str

    def __post_init__(self):
        assert (
            self.points.shape[0] == self.colors.shape[0]
        ), f"{self.points.shape} != {self.colors.shape}"
        assert (
            self.points.shape[0] == self.flows.shape[0]
        ), f"{self.points.shape} != {self.flows.shape}"
        assert (
            np.sum(self.mask) == self.flows.shape[0]
        ), f"Number of valid points in mask ({np.sum(self.mask)}) does not match flow array size ({self.flows.shape[0]})"

    def save(self, parent_folder: Path, idx: int):
        parent_folder.mkdir(parents=True, exist_ok=True)
        feather_path = parent_folder / f"{self.timestamp}.feather"
        save_flow_to_feather(feather_path, self.flows, self.mask)

def save_result(result: SceneFlowData, parent_folder: Path, idx: int):
    result.save(parent_folder, idx)


def render_flows(
    model: EulerFlowModel,
    full_sequence: TorchFullFrameInputSequence,
    base_dataset: AbstractDataset,
    output_folder: Path,
) -> list[SceneFlowData]:
    base_dataset_full_sequence = base_dataset[full_sequence.sequence_idx]

    results: list[SceneFlowData] = []

    # Use torch inference mode
    model.model = model.model.eval()
    with torch.no_grad():
        # No need to predict flows for the last framw
        for idx in tqdm.tqdm(
            range(len(base_dataset_full_sequence)-1), desc="Rendering Flows"
        ):
            torch_query_points = full_sequence.get_global_pc(idx)
            torch_full_mask = full_sequence.get_full_pc_mask(idx) 
            sum = torch_full_mask.sum()
            # A list of TimeSyncedScenewFlowFrame
            scene_flow_frame = base_dataset_full_sequence[idx]

            query_result: ModelFlowResult = model.model(
                torch_query_points,
                idx,
                len(full_sequence),
                QueryDirection.FORWARD,
            )
            flow_np = query_result.flow.detach().cpu().numpy()
            mask_np = torch_full_mask.detach().cpu().numpy()

            pc_frame: SupervisedPointCloudFrame = scene_flow_frame.pc
            if isinstance(pc_frame, ColoredSupervisedPointCloudFrame):
                color_np = pc_frame.colors[pc_frame.mask]
            else:
                color_np = np.ones_like(flow_np)
            pc_np = pc_frame.global_pc.points
            results.append(SceneFlowData(points=pc_np, colors=color_np, flows=flow_np, mask=mask_np, timestamp=scene_flow_frame.log_timestamp))

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
        
        render_flows(model, full_sequence, base_dataset, sequence_output_folder)
            


if __name__ == "__main__":
    main()
