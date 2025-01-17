import numpy as np
import pdb, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv as spconv_core
#from easydict import EasyDict
# import yaml
spconv_core.constants.SPCONV_ALLOW_TF32 = True

import spconv.pytorch as spconv
import time
from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
tv = None
try:
    import cumm.tensorview as tv
except:
    pass

from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.cuda.amp as amp
_TORCH_CUSTOM_FWD = amp.custom_fwd(cast_inputs=torch.float16)
_TORCH_CUSTOM_BWD = amp.custom_bwd

#from typing import List, Tuple, Dict


def conv1x1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv4d(in_planes, out_planes, kernel_size=(1,1,1,3), stride=stride,
                             padding=(0,0,0,1), bias=False, indice_key=indice_key)

def conv3x3x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv4d(in_planes, out_planes, kernel_size=(3,3,3,1), stride=stride,
                             padding=(1,1,1,0), bias=False, indice_key=indice_key)

def conv1x1x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv4d(in_planes, out_planes, kernel_size=(1,1,1,1), stride=stride,
                             padding=0, bias=False, indice_key=indice_key)


def conv3x3x3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv4d(in_planes, out_planes, kernel_size=(3,3,3,3), stride=stride,
                             padding=(1,1,1,1), bias=False, indice_key=indice_key)

norm_cfg = {
    # format: layer_type: (abbreviation, module)
    "BN": ("bn", nn.BatchNorm2d),
    "BN1d": ("bn1d", nn.BatchNorm1d),
    "GN": ("gn", nn.GroupNorm),
}


class SpatioTemporal_Decomposition_Block(nn.Module):
    def __init__(self, in_filters, mid_filters, out_filters, indice_key=None, down_key = None, pooling=False, z_pooling=True, interact=False):
        super(SpatioTemporal_Decomposition_Block, self).__init__()


        self.pooling = pooling

        self.act = nn.LeakyReLU()

        self.spatial_conv_1 = conv3x3x3x1(in_filters, mid_filters, indice_key=indice_key + "bef")
        self.bn_s_1 = nn.BatchNorm1d(mid_filters)

        self.temporal_conv_1 = conv1x1x1x3(in_filters, mid_filters)
        self.bn_t_1 = nn.BatchNorm1d(mid_filters)

        self.fusion_conv_1 = conv1x1x1x1(mid_filters*2+in_filters, mid_filters, indice_key=indice_key + "1D")
        self.bn_fusion_1 = nn.BatchNorm1d(mid_filters)


        self.spatial_conv_2 = conv3x3x3x1(mid_filters, mid_filters, indice_key=indice_key + "bef")
        self.bn_s_2 = nn.BatchNorm1d(mid_filters)

        self.temporal_conv_2 = conv1x1x1x3(mid_filters, mid_filters)
        self.bn_t_2 = nn.BatchNorm1d(mid_filters)

        self.fusion_conv_2 = conv1x1x1x1(mid_filters*3, out_filters, indice_key=indice_key + "1D")
        self.bn_fusion_2 = nn.BatchNorm1d(out_filters)


        if self.pooling:
            if z_pooling == True:
                self.pool = spconv.SparseConv4d(out_filters, out_filters, kernel_size=(2,2,2,1), stride=(2,2,2,1), indice_key=down_key, bias=False)
            else:
                self.pool = spconv.SparseConv4d(out_filters, out_filters, kernel_size=(2,2,1,1), stride=(2,2,1,1), indice_key=down_key, bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        #ST block
        S_feat_1 = self.spatial_conv_1(x)
        S_feat_1 = S_feat_1.replace_feature(self.bn_s_1(S_feat_1.features))
        S_feat_1 = S_feat_1.replace_feature(self.act(S_feat_1.features))

        T_feat_1 = self.temporal_conv_1(x)
        T_feat_1 = T_feat_1.replace_feature(self.bn_t_1(T_feat_1.features))
        T_feat_1 = T_feat_1.replace_feature(self.act(T_feat_1.features))

        ST_feat_1 = x.replace_feature(torch.cat([S_feat_1.features, T_feat_1.features, x.features], 1)) #residual까지 concate

        ST_feat_1 = self.fusion_conv_1(ST_feat_1)
        ST_feat_1 = ST_feat_1.replace_feature(self.bn_fusion_1(ST_feat_1.features))
        ST_feat_1 = ST_feat_1.replace_feature(self.act(ST_feat_1.features))

        #TS block
        S_feat_2 = self.spatial_conv_2(ST_feat_1)
        S_feat_2 = S_feat_2.replace_feature(self.bn_s_2(S_feat_2.features))
        S_feat_2 = S_feat_2.replace_feature(self.act(S_feat_2.features))

        T_feat_2 = self.temporal_conv_2(ST_feat_1)
        T_feat_2 = T_feat_2.replace_feature(self.bn_t_2(T_feat_2.features))
        T_feat_2 = T_feat_2.replace_feature(self.act(T_feat_2.features))

        ST_feat_2 = x.replace_feature(torch.cat([S_feat_2.features, T_feat_2.features, ST_feat_1.features], 1)) #residual까지 concate
        
        ST_feat_2 = self.fusion_conv_2(ST_feat_2)
        ST_feat_2 = ST_feat_2.replace_feature(self.bn_fusion_2(ST_feat_2.features))
        ST_feat_2 = ST_feat_2.replace_feature(self.act(ST_feat_2.features))

        if self.pooling: 
            pooled = self.pool(ST_feat_2)
            return pooled, ST_feat_2
        else:
            return ST_feat_2



class Network_4D(nn.Module):
    def __init__(self, in_channel=16, out_channel=16, model_size = 16):
        super().__init__()

        SpatioTemporal_Block = SpatioTemporal_Decomposition_Block

        self.model_size = model_size
        

        self.STDB_1_1_1 = SpatioTemporal_Block(in_channel, model_size, model_size, indice_key="st1_1", down_key='floor1')
        self.STDB_1_1_2 = SpatioTemporal_Block(model_size, model_size, model_size*2, indice_key="st1_1", down_key='floor1', pooling=True) #512 512 32 -> 256 256 16

        self.STDB_2_1_1 = SpatioTemporal_Block(model_size*2, model_size*2, model_size*2, indice_key="st2_1", down_key='floor2')
        self.STDB_2_1_2 = SpatioTemporal_Block(model_size*2, model_size*2, model_size*4, indice_key="st2_1", down_key='floor2', pooling=True) #256 256 16 -> 128 128 8

        self.STDB_3_1_1 = SpatioTemporal_Block(model_size*4, model_size*4, model_size*4, indice_key="st3_1", down_key='floor3')
        self.STDB_3_1_2 = SpatioTemporal_Block(model_size*4, model_size*4, model_size*4, indice_key="st3_1", down_key='floor3', pooling=True) #128 128 8 -> 64 64 4

        self.STDB_4_1_1 = SpatioTemporal_Block(model_size*4, model_size*4, model_size*4, indice_key="st4_1", down_key='floor4')
        self.STDB_4_1_2 = SpatioTemporal_Block(model_size*4, model_size*4, model_size*4, indice_key="st4_1", down_key='floor4', pooling=True, z_pooling=False) #64 64 4 -> 64 64 4

        self.STDB_5_1_1 = SpatioTemporal_Block(model_size*4, model_size*4, model_size*4, indice_key="st5_1")
        self.STDB_5_1_2 = SpatioTemporal_Block(model_size*4, model_size*4, model_size*4, indice_key="st5_1")
        self.up_subm_5 = spconv.SparseInverseConv4d(model_size*4, model_size*4, kernel_size=(2,2,1,1), indice_key='floor4', bias=False) #zpooling false

        self.STDB_4_2_1 = SpatioTemporal_Block(model_size*8, model_size*8, model_size*4, indice_key="st4_2")
        self.up_subm_4 = spconv.SparseInverseConv4d(model_size*4, model_size*4, kernel_size=(2,2,2,1), indice_key='floor3', bias=False)

        self.STDB_3_2_1 = SpatioTemporal_Block(model_size*8, model_size*8, model_size*4, indice_key="st3_2")
        self.up_subm_3 = spconv.SparseInverseConv4d(model_size*4, model_size*4, kernel_size=(2,2,2,1), indice_key='floor2', bias=False)

        self.STDB_2_2_1 = SpatioTemporal_Block(model_size*8, model_size*4, model_size*4, indice_key="st_2_2")
        self.up_subm_2 = spconv.SparseInverseConv4d(model_size*4, model_size*2, kernel_size=(2,2,2,1), indice_key='floor1', bias=False)

        self.STDB_1_2_1 = SpatioTemporal_Block(model_size*4, model_size*2, out_channel, indice_key="st_1_2")
        

    def forward(self, sp_tensor):

        sp_tensor = self.STDB_1_1_1(sp_tensor)
        down_2, skip_1 = self.STDB_1_1_2(sp_tensor)

        down_2 = self.STDB_2_1_1(down_2)
        down_3, skip_2 = self.STDB_2_1_2(down_2)

        down_3 = self.STDB_3_1_1(down_3)
        down_4, skip_3 = self.STDB_3_1_2(down_3)

        down_4 = self.STDB_4_1_1(down_4)
        down_5, skip_4 = self.STDB_4_1_2(down_4)

        down_5 = self.STDB_5_1_1(down_5)
        down_5 = self.STDB_5_1_2(down_5)

        up_4 = self.up_subm_5(down_5)
        up_4 = up_4.replace_feature(torch.cat((up_4.features, skip_4.features), 1))
        up_4 = self.STDB_4_2_1(up_4)

        up_3 = self.up_subm_4(up_4)
        up_3 = up_3.replace_feature(torch.cat((up_3.features, skip_3.features), 1))
        up_3 = self.STDB_3_2_1(up_3)

        up_2 = self.up_subm_3(up_3)
        up_2 = up_2.replace_feature(torch.cat((up_2.features, skip_2.features), 1))
        up_2 = self.STDB_2_2_1(up_2)

        up_1 = self.up_subm_2(up_2)
        up_1 = up_1.replace_feature(torch.cat((up_1.features, skip_1.features), 1))
        up_1 = self.STDB_1_2_1(up_1)

        return up_1
    


