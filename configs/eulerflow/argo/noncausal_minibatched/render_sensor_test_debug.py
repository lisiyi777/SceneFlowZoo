has_labels = False

SEQUENCE_LENGTH = 150

test_dataset_root = "/efs/argoverse2/test/"
save_output_folder = "/efs/argoverse_sensor_euler_depth18_feather"

model = dict(
    name="EulerFlowDepth18OptimizationLoop",
    args=dict(
        save_flow_every=30,
        minibatch_size=5,
        speed_threshold=60.0 / 10.0,
        lr=0.00008,
        epochs=1000,
        pc_target_type="lidar",
        pc_loss_type="truncated_kd_tree_forward_backward",
    ),
)

train_dataset = dict(
    name="TorchFullFrameDataset",
    args=dict(
        dataset_name="Argoverse2NonCausalSceneFlow",
        root_dir=test_dataset_root,
        load_flow=False,
        with_ground=False,
        with_rgb=False,
        eval_type="bucketed_epe",
        eval_args=dict(),
        subsequence_length=SEQUENCE_LENGTH,
        range_crop_type="ego",  # Ensures that the range is cropped to the ego vehicle, so points are not chopped off if the ego vehicle is driving large distances.
        use_cache=False,
    ),
)


train_dataloader = dict(args=dict(batch_size=1, num_workers=0, shuffle=False, pin_memory=True))


test_dataset = train_dataset.copy()
test_dataloader = train_dataloader.copy()



