_base_ = "./bucketed_nsfp_distillation_1x.py"

train_sequence_dir = "/efs/argoverse2/argoverse2_mini_debug/val/"
test_dataset_root = "/efs/argoverse2/argoverse2_mini_debug/val/"

save_output_folder = "./argoverse2_mini_debug/val_flow4d_rewrite_api/"

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
        eval_args=dict(output_path="eval_results/bucketed_epe_mini_debug/supervised_rewrite_api/"),
    )
)

# Limit batch size to 8 to fit on 24GB RTX3090
train_dataloader = dict(args=dict(batch_size=4, num_workers=0, shuffle=False, pin_memory=True))
test_dataloader = dict(args=dict(batch_size=4, num_workers=0, shuffle=False, pin_memory=True))
validate_every = None
