_base_ = "./test.py"

save_output_folder = "/efs/argoverse2/test_eulerflow_perf/"

model = dict(
    args=dict(
        epochs=1,
    ),
)


test_dataset = dict(
    args=dict(log_subset=["af8471e6-6780-3df2-bc6a-1982a4b1b437"], subsequence_length=156)
)
