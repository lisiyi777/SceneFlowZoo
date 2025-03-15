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



# ref from Flow4D loss function deflowLoss()
class DeFlowLoss(FastFlow3DBaseLoss):
    def __init__(self):
        super().__init__()

    def _deflow_loss(
        self,
        input_batch: list[TorchFullFrameInputSequence],
        model_res: list[TorchFullFrameOutputSequence],
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
        input_batch: list[TorchFullFrameInputSequence],
        model_results: list[TorchFullFrameOutputSequence],
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
        loss_fn: FastFlow3DBaseLoss = DeFlowLoss(),
        # loss_fn: FastFlow3DBaseLoss = FastFlow3DBucketedLoaderLoss(),
    ) -> None:
        super().__init__()

        point_output_ch = 16
        voxel_output_ch = 16
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
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
        # for s in batched_sequence:
        #     mask = s.get_full_pc_mask(0)
        #     sum = mask.sum()
        #     print("sequence_timestamp: ", s.sequence_timestamp)
        batch_dict = self._convert_to_input_dict(batched_sequence)
        model_res = self._model_forward(batch_dict)
        output_sequences = self._convert_output_dict(model_res, batched_sequence)
        return output_sequences


    def _convert_to_input_dict(
        self,
        batched_sequence: List[TorchFullFrameInputSequence],
    ) -> dict:
        def pad_with_nan(pc):
            return torch.nn.utils.rnn.pad_sequence(pc, batch_first=True, padding_value=torch.nan) # pad to the longest valid sequence length
        return {
            "pc0s": pad_with_nan([seq.get_global_pc(-2) for seq in batched_sequence]),
            "pc1s": pad_with_nan([seq.get_global_pc(-1) for seq in batched_sequence]),
            **{
                f"pc_m{i}": pad_with_nan([seq.get_global_pc(-(i + 2)) for seq in batched_sequence])
                for i in range(1, self.SEQUENCE_LENGTH - 1)
            },
            "pc0_transforms": [seq.get_pc_transform_matrices(-2) for seq in batched_sequence]
        }


    def _convert_output_dict(
        self,
        model_res: dict,
        batched_sequence: List[TorchFullFrameInputSequence],
    ) -> List[TorchFullFrameOutputSequence]:
        batch_size = len(next(iter(model_res.values())))

        batch_output = []
        for batch_id in range(batch_size):            
            ego_full_flows = self._convert_to_full_flow(model_res['flow'][batch_id], batched_sequence[batch_id].get_full_pc_mask(-2))
            batch_output.append(
                TorchFullFrameOutputSequence(
                ego_flows=ego_full_flows.unsqueeze(0),
                valid_flow_mask=batched_sequence[batch_id].get_full_pc_mask(-2).unsqueeze(0),
                )
            )
        return batch_output

    def _convert_to_full_flow(self, valid_flows: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        full_flows = torch.zeros((valid_mask.shape[0], 3), device=valid_flows.device)
        full_flows[valid_mask] = valid_flows
        return full_flows

    def _model_forward(self, pcs_dict) -> dict:
        batch_sizes = len(pcs_dict["pc0s"])
        pc0_transforms = pcs_dict["pc0_transforms"]

        dict_4d = self.embedder_4D(pcs_dict)
        pc01_tesnor_4d = dict_4d['4d_tensor']
        pc0_3dvoxel_infos_lst =dict_4d['pc0_3dvoxel_infos_lst']
        pc0_point_feats_lst =dict_4d['pc0_point_feats_lst'] #?
        pc0_num_voxels = dict_4d['pc0_mum_voxels']

        pc_all_output_4d = self.network_4D(pc01_tesnor_4d) #all = past, current, next 다 합친것

        pc0_last = self.seperate_feat(pc_all_output_4d)
        assert pc0_last.features.shape[0] == pc0_num_voxels, 'voxel number mismatch'

        flows = self.pointhead_3D(pc0_last, pc0_3dvoxel_infos_lst, pc0_point_feats_lst)

        pc0_points_lst = [e["points"] for e in pc0_3dvoxel_infos_lst] 
        pc0_valid_point_idxes = [e["point_idxes"] for e in pc0_3dvoxel_infos_lst] 

        ego_flows = []
        for batch_id in range(batch_sizes):
            pc0_sensor_to_ego, pc0_ego_to_global = pc0_transforms[batch_id]
            pc0_valid_mask = pc0_valid_point_idxes[batch_id]
            ego_flow = self.global_to_ego_flow(pcs_dict["pc0s"][batch_id][pc0_valid_mask], flows[batch_id], pc0_ego_to_global)
            ego_flows.append(ego_flow)

        # flows = torch.stack(ego_flows, dim=0) 
        model_res = {
            "flow": ego_flows, 
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
