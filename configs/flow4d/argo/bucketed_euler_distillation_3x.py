_base_ = "./bucketed_supervised.py"


train_sequence_dir = ["/bigdata/argoverse_sensor/train/",
                    "/bigdata/argoverse_lidar/train"
                        ]
train_flow_data_dir = ["/bigdata/argoverse_sensor/train_euler_depth18_feather",
                       "/bigdata/argoverse_lidar/train_euler_depth18_feather"
                       ]
test_dataset_root = "/efs/argoverse2/val/"

save_output_folder = "bigdata/euler_distillation/val_flow4d_3x/"

epochs = 15
learning_rate = 1e-4


train_dataset = dict(args=dict(root_dir=train_sequence_dir, flow_data_path=train_flow_data_dir, use_gt_flow=False))
test_dataset = dict(
    args=dict(
        root_dir=test_dataset_root,
        eval_args=dict(output_path="eval_results/bucketed_epe_distillation/distillation_3x/"),
    )
)

train_dataloader = dict(args=dict(batch_size=4, num_workers=4, shuffle=True, pin_memory=True))
test_dataloader = dict(args=dict(batch_size=4, num_workers=4, shuffle=False, pin_memory=True))
validate_every = None
