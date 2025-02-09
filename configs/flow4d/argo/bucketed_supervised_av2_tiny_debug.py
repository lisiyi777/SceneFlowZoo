_base_ = "./bucketed_nsfp_distillation_1x.py"

train_sequence_dir = "/efs/argoverse2_smaller/val/"
test_dataset_root = "/efs/argoverse2_smaller/val/"

# save_output_folder = "/efs/argoverse2_tiny/val_flow4d_rewrite_api/"
save_output_folder = "./argoverse2_smaller/val_flow4d_rewrite_api/"

epochs = 50

train_dataset = dict(args=dict(root_dir=train_sequence_dir, use_gt_flow=True))
test_dataset = dict(
    args=dict(
        root_dir=test_dataset_root,
        eval_args=dict(output_path="eval_results/bucketed_epe_smaller/supervised_rewrite_api/"),
    )
)

# Limit batch size to 8 to fit on 24GB RTX3090
test_dataloader = dict(args=dict(batch_size=1, num_workers=0, shuffle=False, pin_memory=True))
validate_every = None
