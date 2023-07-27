_base_ = "./nsfp_distilatation_3x.py"


POINT_CLOUD_RANGE = (-51.2, -51.2, -3, 51.2, 51.2, 3)
VOXEL_SIZE = (0.1, 0.1, POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2])
PSEUDO_IMAGE_DIMS = (1024, 1024)


model = dict(name="FastFlow3D",
             args=dict(VOXEL_SIZE=VOXEL_SIZE,
                       PSEUDO_IMAGE_DIMS=PSEUDO_IMAGE_DIMS,
                       FEATURE_CHANNELS=64, 
                       xl_backbone=True))
