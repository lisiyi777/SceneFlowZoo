_base_ = "render_sensor_val.py"

test_dataset_root = "/bigdata/argoverse2/train/"

train_dataset = dict(args=dict(root_dir=test_dataset_root))
train_dataloader = dict(args=dict(batch_size=1, num_workers=0, shuffle=False, pin_memory=True))

test_dataset = train_dataset.copy()
test_dataloader = train_dataloader.copy()
