has_labels = False

SEQUENCE_LENGTH = 5

test_dataset_root = "/efs/argoverse2/train/"
save_output_folder = "/bigdata/argoverse2/argoverse_sensor_RGB"


train_dataset = dict(
    name="TorchFullFrameDataset",
    args=dict(
        dataset_name="Argoverse2CausalSceneFlow",
        root_dir=test_dataset_root,
        with_ground=False,
        use_gt_flow=True, 
        with_rgb=True,
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

