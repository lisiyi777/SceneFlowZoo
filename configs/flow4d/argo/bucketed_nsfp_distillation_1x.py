POINT_CLOUD_RANGE = (-51.2, -51.2, -2.2, 51.2, 51.2, 4.2)
VOXEL_SIZE = (0.2, 0.2, 0.2)
PSEUDO_IMAGE_DIMS = (512, 512)

epochs = 50
learning_rate = 1e-4
save_every = 500
validate_every = 500
gradient_clip_val = 5.0

SEQUENCE_LENGTH = 5
model = dict(
    name="Flow4D",
    args=dict(
        VOXEL_SIZE=VOXEL_SIZE,
        PSEUDO_IMAGE_DIMS=PSEUDO_IMAGE_DIMS,
        POINT_CLOUD_RANGE=POINT_CLOUD_RANGE,
        FEATURE_CHANNELS=32,
        SEQUENCE_LENGTH=SEQUENCE_LENGTH,
    ),
)

######## TEST DATASET ########

test_dataset_root = "/efs/argoverse2/val/"

test_dataset = dict(
    name="TorchFullFrameDataset",
    args=dict(
        dataset_name="Argoverse2CausalSceneFlow",
        root_dir=test_dataset_root,
        subsequence_length=5,
        with_ground=False,
        with_rgb=False,
        eval_type="bucketed_epe",
        expected_camera_shape=(194, 256, 3),
        # point_cloud_range=None,
        eval_args=dict(output_path="eval_results/bucketed_epe/nsfp_distillation_1x/"),
    ),
)

test_dataloader = dict(args=dict(batch_size=4, num_workers=4, shuffle=False, pin_memory=True))

######## TRAIN DATASET ########

train_sequence_dir = "/efs/argoverse2/train/"

train_dataset = dict(
    name="TorchFullFrameDataset",
    args=dict(
        dataset_name="Argoverse2CausalSceneFlow",
        root_dir=train_sequence_dir,
        subsequence_length=5,
        with_ground=False,
        use_gt_flow=False,
        with_rgb=False,
        eval_type="bucketed_epe",
        expected_camera_shape=(194, 256, 3),
        # point_cloud_range=None,
        eval_args=dict(),
    ),
)

train_dataloader = dict(args=dict(batch_size=4, num_workers=4, shuffle=True, pin_memory=False))
