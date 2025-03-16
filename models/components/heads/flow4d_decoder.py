import torch
import torch.nn as nn
import spconv as spconv_core
#from easydict import EasyDict
spconv_core.constants.SPCONV_ALLOW_TF32 = True

import spconv.pytorch as spconv
import time
from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
tv = None
try:
    import cumm.tensorview as tv
except:
    pass

class Seperate_to_3D(nn.Module):
    def __init__(self, num_frames):
        super(Seperate_to_3D, self).__init__()
        self.num_frames = num_frames
        #self.return_pc1 = return_pc1

    def forward(self, sparse_4D_tensor):

        indices_4d = sparse_4D_tensor.indices
        features_4d = sparse_4D_tensor.features
        
        pc0_time_value = self.num_frames-2

        mask_pc0 = (indices_4d[:, -1] == pc0_time_value)
        
        pc0_indices = indices_4d[mask_pc0][:, :-1] 
        pc0_features = features_4d[mask_pc0]

        pc0_sparse_3D = sparse_4D_tensor.replace_feature(pc0_features)
        pc0_sparse_3D.spatial_shape = sparse_4D_tensor.spatial_shape[:-1]
        pc0_sparse_3D.indices = pc0_indices

        return pc0_sparse_3D

class Point_head(nn.Module):
    def __init__(self, voxel_feat_dim: int = 96, point_feat_dim: int = 32):
        super().__init__()

        self.input_dim = voxel_feat_dim + point_feat_dim

        self.PPmodel_flow = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 3)
        )

    def forward_single(self, voxel_feat, voxel_coords, point_feat):

        voxel_to_point_feat = voxel_feat[:, voxel_coords[:,2], voxel_coords[:,1], voxel_coords[:,0]].T 
        concated_point_feat = torch.cat([voxel_to_point_feat, point_feat],dim=-1)

        flow = self.PPmodel_flow(concated_point_feat)

        return flow

    def forward(self, sparse_tensor, voxelizer_infos, pc0_point_feats_lst): 
        
        voxel_feats = sparse_tensor.dense()

        flow_outputs = []
        batch_idx = 0
        for voxelizer_info in voxelizer_infos:
            voxel_coords = voxelizer_info["voxel_coords"]
            point_feat = pc0_point_feats_lst[batch_idx]
            voxel_feat = voxel_feats[batch_idx, :]
            flow = self.forward_single(voxel_feat, voxel_coords, point_feat)
            batch_idx += 1 
            flow_outputs.append(flow)

        return flow_outputs







