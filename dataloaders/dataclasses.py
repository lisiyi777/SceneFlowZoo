from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from bucketed_scene_flow_eval.datastructures import (
    SE3,
    GroundTruthPointFlow,
    Timestamp,
)

from pointclouds import from_fixed_array


@dataclass
class BucketedSceneFlowItem:
    """
    Class that contains all the data required for computing scene flow of a single sample

    Args:
        dataset_log_id: string of the log id in the dataset
        dataset_idx: the index of the sample in the dataset
        query_timestamp: An int specifying which timestep in the sequence to start the flow query.
        raw_source_pc: Source point cloud for the scene flow problem, shape <N, 3>
        raw_source_pc_mask: Source point cloud mask for the scene flow problem, shape <N>
        raw_target_pc: Target point cloud for the scene flow problem, shape <M, 3>
        raw_target_pc_mask: Target point cloud mask for the scene flow problem, shape <M>
        source_pose: SE3 object for the pose at the source
        target_pose: SE3 object for the pose at the target
        full_percept_pcs_array_stack: Provided for convenience, the full pc instead of just the flow query points, shape <K, PadN, 3>
        full_percept_pose_array_stack: Full pose, shape <K, 4, 4>
        raw_gt_flowed_source_pc: The source point cloud with ground truth flow vectors applied, shape <N, 3>
        raw_gt_flowed_source_pc_mask: The mask for the ground truth flowed source point cloud, shape <N>
        raw_gt_pc_class_mask: The class ID for each point in the point cloud, shape <N>
        gt_trajectories: GroundTruthPointFlow object containing the ground truth trajectories.
    """

    dataset_log_id: str
    dataset_idx: int
    query_timestamp: Timestamp
    raw_source_pc: torch.FloatTensor  # (N, 3)
    raw_source_pc_mask: torch.BoolTensor  # (N, 3)
    raw_target_pc: torch.FloatTensor  # (M, 3)
    raw_target_pc_mask: torch.BoolTensor  # (M, 3)
    source_pose: SE3
    target_pose: SE3
    raw_gt_flowed_source_pc: torch.FloatTensor  # (N, 3)
    raw_gt_flowed_source_pc_mask: torch.BoolTensor  # (N, 3)
    raw_gt_pc_class_mask: torch.LongTensor  # (N,)

    # Included for completeness, these are the full percepts provided by the
    # dataloader rather than just the scene flow query points.
    # This allows for the data loader to e.g. provide more point clouds for accumulation.
    all_percept_pcs_array_stack: torch.FloatTensor  # (K, PadN, 3)
    all_percept_pose_array_stack: torch.FloatTensor  # (K, 4, 4)
    gt_trajectories: GroundTruthPointFlow

    def __post_init__(self):
        # Ensure the point clouds are _ x 3
        assert (
            self.raw_source_pc.shape[1] == 3
        ), f"Raw source pc shape is {self.raw_source_pc.shape}"
        assert (
            self.raw_target_pc.shape[1] == 3
        ), f"Raw target pc shape is {self.raw_target_pc.shape}"

        # Ensure the point cloud masks are boolean and the same length as the point clouds
        assert (
            self.raw_source_pc_mask.dtype == torch.bool
        ), f"Raw source pc mask dtype is {self.raw_source_pc_mask.dtype}"
        assert (
            self.raw_target_pc_mask.dtype == torch.bool
        ), f"Raw target pc mask dtype is {self.raw_target_pc_mask.dtype}"
        assert (
            self.raw_source_pc_mask.shape == self.raw_source_pc.shape[:1]
        ), f"Raw source pc mask shape is {self.raw_source_pc_mask.shape}"
        assert (
            self.raw_target_pc_mask.shape == self.raw_target_pc.shape[:1]
        ), f"Raw target pc mask shape is {self.raw_target_pc_mask.shape}"

        # Ensure the poses are SE3 objects
        assert isinstance(self.source_pose, SE3), f"Source pose is {self.source_pose}"
        assert isinstance(self.target_pose, SE3), f"Target pose is {self.target_pose}"

        # Ensure the ground truth point cloud is _ x 3
        assert (
            self.raw_gt_flowed_source_pc.shape[1] == 3
        ), f"Raw gt flowed source pc shape is {self.raw_gt_flowed_source_pc.shape}"

        # Ensure the ground truth point cloud mask is boolean and the same length as the point clouds
        assert (
            self.raw_gt_flowed_source_pc_mask.dtype == torch.bool
        ), f"Raw gt flowed source pc mask dtype is {self.raw_gt_flowed_source_pc_mask.dtype}"
        assert (
            self.raw_gt_flowed_source_pc_mask.shape == self.raw_gt_flowed_source_pc.shape[:1]
        ), f"Raw gt flowed source pc mask shape is {self.raw_gt_flowed_source_pc_mask.shape}"

        # Ensure the ground truth point cloud class mask is the same length as the point clouds
        assert (
            self.raw_gt_pc_class_mask.shape == self.raw_gt_flowed_source_pc.shape[:1]
        ), f"Raw gt pc class mask shape is {self.raw_gt_pc_class_mask.shape}"

    @property
    def source_pc(self) -> torch.FloatTensor:
        return self.raw_source_pc[self.raw_source_pc_mask]

    @property
    def target_pc(self) -> torch.FloatTensor:
        return self.raw_target_pc[self.raw_target_pc_mask]

    @property
    def gt_flowed_source_pc(self) -> torch.FloatTensor:
        return self.raw_gt_flowed_source_pc[self.raw_gt_flowed_source_pc_mask]

    @property
    def gt_pc_class_mask(self) -> torch.LongTensor:
        return self.raw_gt_pc_class_mask[self.raw_gt_flowed_source_pc_mask]

    def to(self, device: str) -> None:
        """
        Copy tensors in this batch to the target device.

        Args:
            device: the string (and optional ordinal) used to construct the device object ex. 'cuda:0'
        """
        self.raw_source_pc = self.raw_source_pc.to(device)
        self.raw_source_pc_mask = self.raw_source_pc_mask.to(device)
        self.raw_target_pc = self.raw_target_pc.to(device)
        self.raw_target_pc_mask = self.raw_target_pc_mask.to(device)
        self.all_percept_pcs_array_stack = self.all_percept_pcs_array_stack.to(device)
        self.all_percept_pose_array_stack = self.all_percept_pose_array_stack.to(device)
        self.raw_gt_flowed_source_pc = self.raw_gt_flowed_source_pc.to(device)
        self.raw_gt_flowed_source_pc_mask = self.raw_gt_flowed_source_pc_mask.to(device)
        self.raw_gt_pc_class_mask = self.raw_gt_pc_class_mask.to(device)
        return self

    def full_percept(self, idx: int) -> Tuple[np.ndarray, SE3]:
        return from_fixed_array(self.all_percept_pcs_array_stack[idx]), SE3.from_array(
            self.all_percept_pose_array_stack[idx]
        )

    def full_percepts(self) -> List[Tuple[np.ndarray, SE3]]:
        return [self.full_percept(idx) for idx in range(len(self.all_percept_pcs_array_stack))]


# @dataclass
# class BucketedSceneFlowBatchOutput():
#     """
#     A standardized set of outputs for Bucketed Scene Flow evaluation.
#     In this dataclass, N is the number of points in pc0, M is for pc1, and all lists have len = batch size.

#     Args:
#         flow: A list of <N, 3> tensors containing the flow for each point.
#         pc0_points_list: A list of <N, 3> tensors containing pc0.
#         pc0_valid_point_indexes: A list of <N> tensors containing a valid mask for the pointcloud pc0.
#         pc1_points_list: A list of <M, 3> tensors containing pc1.
#         pc1_valid_point_indexes: A list of <M> tensors containing a valid mask for the pointcloud pc1.
#         pc0_warped_points_list: An optional list of <N, 3> tensors containing the points of pc0 with the flow vectors added to them.
#         batch_delta_time: An optional float of the amount of time to compute flow for the batch.
#     """
#     flow: List[torch.FloatTensor]
#     pc0_points_list: List[torch.FloatTensor]
#     pc0_valid_point_indexes: List[torch.LongTensor]
#     pc1_points_list: List[torch.FloatTensor]
#     pc1_valid_point_indexes: List[torch.LongTensor]
#     pc0_warped_points_list: Optional[List[torch.FloatTensor]]
#     batch_delta_time: Optional[float]


@dataclass
class BucketedSceneFlowOutputItem:
    """
    A standardized set of outputs for Bucketed Scene Flow evaluation.
    In this dataclass, N is the number of points in pc0, M is for pc1.

    Args:
        flow: A <N, 3> tensor containing the flow for each point.
        pc0_points: A <N, 3> tensor containing pc0.
        pc0_valid_point_indexes: A <N> tensor containing a valid mask for the pointcloud pc0.
        pc1_points: A <M, 3> tensor containing pc1.
        pc1_valid_point_indexes: A <M> tensor containing a valid mask for the pointcloud pc1.
        pc0_warped_points: A <N, 3> tensor containing the points of pc0 with the flow vectors added to them.
    """

    flow: torch.FloatTensor
    pc0_points: torch.FloatTensor
    pc0_valid_point_indexes: torch.LongTensor
    pc1_points: torch.FloatTensor
    pc1_valid_point_indexes: torch.LongTensor
    pc0_warped_points: torch.FloatTensor

    def to(self, device: str) -> None:
        """
        Copy tensors in this batch to the target device.

        Args:
            device: the string (and optional ordinal) used to construct the device object ex. 'cuda:0'
        """
        self.flow = self.flow.to(device)
        self.pc0_points = self.target_pc.to(device)
        self.pc0_valid_point_indexes = self.pc0_valid_point_indexes.to(device)
        self.pc1_points = self.pc1_points.to(device)
        self.pc1_valid_point_indexes = self.pc1_valid_point_indexes.to(device)
        self.pc0_warped_points = self.pc0_warped_points.to(device)
        return self