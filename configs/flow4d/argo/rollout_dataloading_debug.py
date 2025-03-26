_base_ = "./bucketed_supervised.py"

train_sequence_dir = "/efs/argoverse2_mini_debug/val/"
train_flow_data_dir = "/efs/argoverse2_mini_debug/val_rollout_feather"

test_dataset_root = "/efs/argoverse2_mini_debug/val/"

save_output_folder = "bigdata/argoverse2_mini_debug/val_flow4d_rollouts/"

epochs = 15
learning_rate = 1e-4
seq_length=5
rollout_steps=1

model = dict(
    args=dict(
        FEATURE_CHANNELS=32,
        rollout_steps=rollout_steps,
    ),
)



train_dataset = dict( 
    args=dict(
        root_dir=train_sequence_dir, 
        flow_data_path=train_flow_data_dir, 
        use_gt_flow=False, 
        rollout_steps=rollout_steps,
        subsequence_length=seq_length+rollout_steps-1,
        ),
    )
test_dataset = dict(
    args=dict(
        root_dir=test_dataset_root,
        subsequence_length=seq_length,
        eval_args=dict(output_path="eval_results/bucketed_epe_mini_debug/supervised_rollouts/"),
    )
)

train_dataloader = dict(args=dict(batch_size=1, num_workers=0, shuffle=False, pin_memory=True))
test_dataloader = dict(args=dict(batch_size=1, num_workers=0, shuffle=False, pin_memory=True))
validate_every = None
