_base_ = "./bucketed_supervised.py"

train_sequence_dir = "/efs/argoverse2/argoverse2_mini_debug/val/"
test_dataset_root = "/efs/argoverse2/argoverse2_mini_debug/val/"

save_output_folder = "bigdata/argoverse2_mini_debug/val_flow4d/"

epochs = 15
learning_rate = 1e-4

model = dict(
    args=dict(
        FEATURE_CHANNELS=32,
    ),
)

train_dataset = dict(args=dict(root_dir=train_sequence_dir, use_gt_flow=True))
test_dataset = dict(
    args=dict(
        root_dir=test_dataset_root,
        eval_args=dict(output_path="eval_results/bucketed_epe_mini_debug/supervised/"),
    )
)

train_dataloader = dict(args=dict(batch_size=1, num_workers=0, shuffle=False, pin_memory=True))
test_dataloader = dict(args=dict(batch_size=1, num_workers=0, shuffle=False, pin_memory=True))
validate_every = None
