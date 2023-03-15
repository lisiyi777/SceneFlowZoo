_base_ = "./unsupervised_sensor_train.py"

test_sequence_dir = "/efs/waymo_open_preprocessed/val/"
flow_save_folder = "/efs/waymo_open_preprocessed/val_nsfp_flow/"

model = dict(args=dict(flow_save_folder=flow_save_folder))

test_loader = dict(args=dict(sequence_dir=test_sequence_dir))
