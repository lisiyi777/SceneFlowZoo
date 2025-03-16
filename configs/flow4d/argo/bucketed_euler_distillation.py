_base_ = "./bucketed_nsfp_distillation_1x.py"

train_dataset = dict(args=dict(use_gt_flow=True))
test_dataset = dict(
    args=dict(
              eval_args=dict(output_path = "eval_results/bucketed_epe/supervised/")))

train_sequence_dir = "/efs/argoverse2/train/"
train_flow_data_dir = "/efs/argoverse_sensor/train_euler_depth18_feather"
test_dataset_root = "/efs/argoverse2/val/"


save_output_folder = "./euler_distillation/val_flow4d/"

epochs = 15
learning_rate = 1e-4


train_dataset = dict(args=dict(root_dir=train_sequence_dir, flow_data_path=train_flow_data_dir, use_gt_flow=False))
test_dataset = dict(
    args=dict(
        root_dir=test_dataset_root,
        eval_args=dict(output_path="eval_results/bucketed_epe_sensor/distillation_train/"),
    )
)

# Limit batch size to 8 to fit on 24GB RTX3090
train_dataloader = dict(args=dict(batch_size=4, num_workers=4, shuffle=True, pin_memory=True))
test_dataloader = dict(args=dict(batch_size=4, num_workers=4, shuffle=False, pin_memory=True))
validate_every = None
