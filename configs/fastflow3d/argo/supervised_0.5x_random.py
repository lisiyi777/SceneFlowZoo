_base_ = "./supervised.py"

epochs = 100
check_val_every_n_epoch = 1
validate_every = None
dataset = dict(args=dict(subset_fraction=0.5, subset_mode='random'))