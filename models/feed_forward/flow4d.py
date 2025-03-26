"""
Copied with modification from: https://github.com/dgist-cvlab/Flow4D
"""
from typing import List

import torch
import torch.nn as nn
import dztimer
from .fast_flow_3d import (
    FastFlow3D,
    FastFlow3DHeadType,
    FastFlow3DBackboneType,
    FastFlow3DBaseLoss,
    FastFlow3DBucketedLoaderLoss,
)
from models.components.backbones import Network_4D
from models.components.heads import Seperate_to_3D, Point_head
from models.components.embedders import DynamicEmbedder_4D

from dataloaders import TorchFullFrameInputSequence, TorchFullFrameOutputSequence
from models.components.backbones import FastFlowUNet, FastFlowUNetXL
from models.components.embedders import DynamicEmbedder
from models.components.heads import FastFlowDecoder, FastFlowDecoderStepDown, ConvGRUDecoder
from pointclouds.losses import warped_pc_loss
from models.base_models import BaseTorchModel, ForwardMode
import enum
from pytorch_lightning.loggers import Logger
from abc import ABC, abstractmethod
import pickle


class DeFlowBaseLoss(ABC):
    @abstractmethod
    def __call__(
        self,
        input_batch: list[TorchFullFrameInputSequence],
        model_results: List[TorchFullFrameOutputSequence],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError()

# ref from Flow4D loss function deflowLoss()
class DeFlowLoss(DeFlowBaseLoss):
    def __init__(self):
        super().__init__()

    def _deflow_loss(
        self,
        input_batch: List[TorchFullFrameInputSequence],
        model_res: List[TorchFullFrameOutputSequence],
    ):
        total_loss = 0.

        for input_item, output_item in zip(input_batch, model_res):
            source_idx = len(input_item) - 2
            gt = input_item.get_full_ego_pc_gt_flowed(source_idx) - input_item.get_full_ego_pc(source_idx)
            pred = output_item.get_full_ego_flow(0)

            speed = gt.norm(dim=1, p=2) / 0.1

            pts_loss = torch.linalg.vector_norm(pred - gt, dim=-1)

            weight_loss = 0.
            speed_0_4 = pts_loss[speed < 0.4].mean()
            speed_mid = pts_loss[(speed >= 0.4) & (speed <= 1.0)].mean()
            speed_1_0 = pts_loss[speed > 1.0].mean()

            if ~speed_1_0.isnan():
                weight_loss += speed_1_0
            if ~speed_0_4.isnan():
                weight_loss += speed_0_4
            if ~speed_mid.isnan():
                weight_loss += speed_mid

            total_loss += weight_loss

        return total_loss

    def __call__(
        self,
        input_batch: List[TorchFullFrameInputSequence],
        model_results: List[TorchFullFrameOutputSequence],
    ) -> dict[str, torch.Tensor]:
        loss = self._deflow_loss(input_batch, model_results)
        return {"loss": loss}

class DeFlowRolloutLoss(DeFlowBaseLoss):
    def __init__(self):
        super().__init__()

    def _deflow_loss(
        self,
        input_batch: List[TorchFullFrameInputSequence],
        model_res: List[TorchFullFrameOutputSequence],
    ):
        total_loss = 0.0

        for input_item, output_item in zip(input_batch, model_res):
            sequence_len = len(input_item)
            rollout_steps = len(output_item)
            source_idx = sequence_len - rollout_steps - 1

            assert source_idx >= 0, f"Invalid source_idx={source_idx}"

            for t in range(rollout_steps):
                pred = output_item.get_full_ego_flow(t)
                if t == 0:
                    gt = input_item.get_full_ego_pc_gt_flowed(source_idx) - input_item.get_full_ego_pc(source_idx)
                else:
                    gt = input_item.get_full_ego_pc_gt_multi_step_flowed(source_idx, t-1) - input_item.get_full_ego_pc(source_idx)

                assert pred.shape == gt.shape, f"Shape mismatch: pred={pred.shape}, gt={gt.shape}"

                speed = gt.norm(dim=1, p=2) / 0.1
                pts_loss = torch.linalg.vector_norm(pred - gt, dim=-1)

                # Weighted loss by speed buckets
                weight_loss = 0.0
                speed_0_4 = pts_loss[speed < 0.4].mean()
                speed_mid = pts_loss[(speed >= 0.4) & (speed <= 1.0)].mean()
                speed_1_0 = pts_loss[speed > 1.0].mean()

                if not torch.isnan(speed_1_0):
                    weight_loss += speed_1_0
                if not torch.isnan(speed_0_4):
                    weight_loss += speed_0_4
                if not torch.isnan(speed_mid):
                    weight_loss += speed_mid

                total_loss += weight_loss

        return total_loss

    def __call__(
        self,
        input_batch: List[TorchFullFrameInputSequence],
        model_results: List[TorchFullFrameOutputSequence],
    ) -> dict[str, torch.Tensor]:
        loss = self._deflow_loss(input_batch, model_results)
        return {"loss": loss}

class Flow4D(BaseTorchModel):
    def __init__(
        self,
        VOXEL_SIZE=[0.2, 0.2, 0.2],
        PSEUDO_IMAGE_DIMS=[512, 512],
        POINT_CLOUD_RANGE=[-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
        FEATURE_CHANNELS=32,
        SEQUENCE_LENGTH=5,
        rollout_steps=3,
        loss_fn: DeFlowBaseLoss = DeFlowRolloutLoss(),
        # loss_fn: FastFlow3DBaseLoss = FastFlow3DBucketedLoaderLoss(),
    ) -> None:
        super().__init__()

        point_output_ch = 8
        voxel_output_ch = 8
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        self.rollout_steps=rollout_steps
        self.embedder_4D = DynamicEmbedder_4D(voxel_size=VOXEL_SIZE,
                                        pseudo_image_dims=[PSEUDO_IMAGE_DIMS[0], PSEUDO_IMAGE_DIMS[1], FEATURE_CHANNELS, SEQUENCE_LENGTH], 
                                        point_cloud_range=POINT_CLOUD_RANGE,
                                        feat_channels=point_output_ch)
        
        self.network_4D = Network_4D(in_channel=point_output_ch, out_channel=voxel_output_ch)

        self.seperate_feat = Seperate_to_3D(SEQUENCE_LENGTH)

        self.pointhead_3D = Point_head(voxel_feat_dim=voxel_output_ch, point_feat_dim=point_output_ch)

        self.loss_fn_obj = loss_fn

        self.timer = dztimer.Timing()
        self.timer.start("Total")

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {k[len("model.") :]: v for k, v in ckpt.items() if k.startswith("model.")}
        print("\nLoading... model weight from: ", ckpt_path, "\n")
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(
        self,
        forward_mode: ForwardMode,
        batched_sequence: List[TorchFullFrameInputSequence],
        logger: Logger,
    ) -> List[TorchFullFrameOutputSequence]:
        
        pred_results = []
        full_len = len(batched_sequence[0])
        batch_size = len(batched_sequence)
        
        if forward_mode == ForwardMode.VAL:
            assert full_len == self.SEQUENCE_LENGTH, f"Expected full_len to be {self.SEQUENCE_LENGTH}, but got {full_len}"

            window = [
                [seq.get_global_pc(i) for seq in batched_sequence]
                for i in range(self.SEQUENCE_LENGTH)
            ]
            
            model_res = self._model_forward(window)
            pred_results.append(model_res)

        elif forward_mode == ForwardMode.TRAIN:
            rollout_steps = self.rollout_steps
            assert full_len == self.SEQUENCE_LENGTH + rollout_steps - 1, \
                f"Expected full_len to be {self.SEQUENCE_LENGTH + rollout_steps - 1}, but got {full_len}"

            # Prepare initial window
            window = [
                [seq.get_global_pc(i) for seq in batched_sequence]
                for i in range(self.SEQUENCE_LENGTH)
            ]

            for rollout_step in range(rollout_steps):
                model_res = self._model_forward(window)
                flows = model_res["flow"]
                last_pc_batch = window[-2]

                warped_pc = [last_pc_batch[i] + flows[i] for i in range(batch_size)]

                pred_results.append(model_res)

                # Prepare next window
                if self.SEQUENCE_LENGTH + rollout_step < full_len:
                    next_index = self.SEQUENCE_LENGTH + rollout_step
                    next_frame = [seq.get_global_pc(next_index) for seq in batched_sequence]
                else:
                    break

                # Sliding the window forward
                window = window[1:-1] + [warped_pc, next_frame]

        return self._convert_output_dict(pred_results, batched_sequence)
    

    def _convert_output_dict(
        self,
        pred_results: List[dict],  # rollout_steps × model_res dicts
        batched_sequence: List[TorchFullFrameInputSequence],
    ) -> List[TorchFullFrameOutputSequence]:
        batch_size = len(batched_sequence)
        rollout_steps = len(pred_results)
        outputs = []


        for batch_id in range(batch_size):
            flow_seq = []
            valid_mask_seq = []

            _, ego_to_global = batched_sequence[batch_id].get_pc_transform_matrices(self.SEQUENCE_LENGTH - 2)
            pc0_valid = batched_sequence[batch_id].get_global_pc(self.SEQUENCE_LENGTH - 2)
            pc0_mask = batched_sequence[batch_id].get_full_pc_mask(self.SEQUENCE_LENGTH - 2)

            pc_pred = pc0_valid.clone()

            for step in range(rollout_steps):
                model_res = pred_results[step]
                flow = model_res["flow"][batch_id]  # shape: (N_valid, 3)

                assert flow.shape == pc_pred.shape, \
                    f"[ERROR] Flow shape mismatch at step {step}: flow={flow.shape}, pc_pred={pc_pred.shape}"

                pc_pred = pc_pred + flow

                # Convert to ego frame
                flow_ego = self.global_to_ego_flow(pc0_valid, pc_pred - pc0_valid, ego_to_global)

                # Scatter back into full-sized array
                full_flows = torch.zeros((pc0_mask.shape[0], 3), device=pc0_mask.device)
                full_flows[pc0_mask] = flow_ego

                flow_seq.append(full_flows.unsqueeze(0))
                valid_mask_seq.append(pc0_mask.unsqueeze(0))

            flow_tensor = torch.cat(flow_seq, dim=0)          # (rollout_steps, N, 3)
            valid_mask_tensor = torch.cat(valid_mask_seq, 0)  # (rollout_steps, N)


            outputs.append(
                TorchFullFrameOutputSequence(
                    ego_flows=flow_tensor,
                    valid_flow_mask=valid_mask_tensor,
                )
            )

        return outputs

    def _convert_to_full_flow(self, valid_flows: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        full_flows = torch.zeros((valid_mask.shape[0], 3), device=valid_flows.device)
        full_flows[valid_mask] = valid_flows
        return full_flows

    def _model_forward(self, pcs: List[List[torch.Tensor]]) -> dict:
        """
        Args:
            pcs: List of length SEQUENCE_LENGTH, each item is a list of length batch size, each item is [N_i, 3] tensor of points in global coordinates.

        Returns:
            flows: List (length=batch size) of flow tensors in global coordinates, each of shape [N_0, 3]
        """
        assert len(pcs) == self.SEQUENCE_LENGTH, f"Expected {self.SEQUENCE_LENGTH} frames, got {len(pcs)}"

        def pad_with_nan(pc_list):
            return torch.nn.utils.rnn.pad_sequence(pc_list, batch_first=True, padding_value=torch.nan)

        pc0 = pcs[-2]  # second-to-last frame
        pc1 = pcs[-1]  # last frame
        pc_m = pcs[:-2][::-1]  # past frames in reverse order

        pcs_dict = {
            "pc0s": pad_with_nan(pc0),
            "pc1s": pad_with_nan(pc1),
            **{
                f"pc_m{i}": pad_with_nan(pc) for i, pc in enumerate(pc_m, start=1)
            }
        }

        dict_4d = self.embedder_4D(pcs_dict)
        pc01_tensor_4d = dict_4d['4d_tensor']
        pc0_3dvoxel_infos_lst = dict_4d['pc0_3dvoxel_infos_lst']
        pc0_point_feats_lst = dict_4d['pc0_point_feats_lst']
        pc0_num_voxels = dict_4d['pc0_mum_voxels']

        pc_all_output_4d = self.network_4D(pc01_tensor_4d)
        pc0_last = self.seperate_feat(pc_all_output_4d)
        assert pc0_last.features.shape[0] == pc0_num_voxels, 'voxel number mismatch'

        flows = self.pointhead_3D(pc0_last, pc0_3dvoxel_infos_lst, pc0_point_feats_lst)
        pc0_points_lst = [e["points"] for e in pc0_3dvoxel_infos_lst] 
        pc0_valid_point_idxes = [e["point_idxes"] for e in pc0_3dvoxel_infos_lst] 

        model_res = {
            "flow": flows, 
            "pc0_valid_point_idxes": pc0_valid_point_idxes, 
            "pc0_points_lst": pc0_points_lst, 
        }
        
        return model_res


    def loss_fn(
        self,
        input_batch: List[TorchFullFrameInputSequence],
        model_res: List[TorchFullFrameOutputSequence],
    ) -> dict[str, torch.Tensor]:
        return self.loss_fn_obj(input_batch, model_res)
    

    