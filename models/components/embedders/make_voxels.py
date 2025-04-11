import torch
import torch.nn as nn
from mmcv.ops import Voxelization
from typing import List, Tuple


class HardVoxelizer(nn.Module):

    def __init__(self, voxel_size, point_cloud_range,
                 max_points_per_voxel: int):
        super().__init__()
        assert max_points_per_voxel > 0, f"max_points_per_voxel must be > 0, got {max_points_per_voxel}"

        self.voxelizer = Voxelization(voxel_size,
                                      point_cloud_range,
                                      max_points_per_voxel,
                                      deterministic=False)

    def forward(self, points: torch.Tensor):
        assert isinstance(
            points,
            torch.Tensor), f"points must be a torch.Tensor, got {type(points)}"
        not_nan_mask = ~torch.isnan(points).any(dim=2)
        return {"voxel_coords": self.voxelizer(points[not_nan_mask])}


class DynamicVoxelizer(nn.Module):

    def __init__(self, voxel_size, point_cloud_range):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.voxelizer = Voxelization(voxel_size, point_cloud_range, max_num_points=-1)

    def _get_point_offsets(self, xyz_points: torch.Tensor, voxel_coords: torch.Tensor):
        point_cloud_range = torch.tensor(self.point_cloud_range, dtype=xyz_points.dtype, device=xyz_points.device)
        min_point = point_cloud_range[:3]
        voxel_size = torch.tensor(self.voxel_size, dtype=xyz_points.dtype, device=xyz_points.device)

        # voxel_coords are Z, Y, X -> convert to X, Y, Z
        voxel_coords = voxel_coords[:, [2, 1, 0]]

        voxel_centers = voxel_coords * voxel_size + min_point + voxel_size / 2

        return xyz_points - voxel_centers

    def forward(self, points: List[torch.Tensor], frame_key=None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        batch_results = []

        for batch_idx in range(len(points)):
            batch_points = points[batch_idx]                     # (N, input_dim)
            xyz_points = batch_points[:, :3]                     # only xyz used for voxelization
            extra_feats = batch_points[:, 3:] if batch_points.shape[1] > 3 else None

            valid_point_idxes = torch.arange(batch_points.shape[0], device=batch_points.device)
            not_nan_mask = ~torch.isnan(xyz_points).any(dim=1)   # ✅ only check xyz for nan
            xyz_points = xyz_points[not_nan_mask]
            valid_point_idxes = valid_point_idxes[not_nan_mask]

            if extra_feats is not None:
                extra_feats = extra_feats[not_nan_mask]

            if frame_key == 'pc0s':
                # Only clip xyz
                point_cloud_range = torch.tensor(self.point_cloud_range, dtype=xyz_points.dtype, device=xyz_points.device)
                xyz_points = torch.clamp(xyz_points, min=point_cloud_range[:3]+1e-5, max=point_cloud_range[3:]-1e-5)

            batch_voxel_coords = self.voxelizer(xyz_points)

            # Remove points that are outside of voxel range
            batch_voxel_coords_mask = (batch_voxel_coords != -1).all(dim=1)
            valid_batch_voxel_coords = batch_voxel_coords[batch_voxel_coords_mask]
            xyz_points = xyz_points[batch_voxel_coords_mask]
            valid_point_idxes = valid_point_idxes[batch_voxel_coords_mask]

            if extra_feats is not None:
                extra_feats = extra_feats[batch_voxel_coords_mask]
                valid_batch_points = torch.cat([xyz_points, extra_feats], dim=1)   # ✅ re-concatenate
            else:
                valid_batch_points = xyz_points

            point_offsets = self._get_point_offsets(xyz_points, valid_batch_voxel_coords)

            result_dict = {
                "points": valid_batch_points,      # ✅ keep the full points (xyz + extra_feats)
                "voxel_coords": valid_batch_voxel_coords,
                "point_idxes": valid_point_idxes,
                "point_offsets": point_offsets,
            }

            batch_results.append(result_dict)

        return batch_results
