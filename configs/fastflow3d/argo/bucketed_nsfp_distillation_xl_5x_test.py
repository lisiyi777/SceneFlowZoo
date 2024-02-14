_base_ = "./bucketed_nsfp_distillation_xl_5x.py"

has_labels = False

test_dataset_root = "/efs/argoverse2/test/"

test_dataset = dict(
    args=dict(
        root_dir=test_dataset_root,
        eval_args=dict(output_path="eval_results/bucketed_epe/nsfp_distillation_xl_5x_test/"),
    )
)
