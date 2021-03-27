import datetime
import os
import torch

curr_date = datetime.datetime.now().strftime("%b%d_%H-%M-%S")

grid_size = 7
class_amount = 20
bbox_pred_amount = 2

debug = False
print_freq = 1000
device = torch.device("cuda:0")
loader_workers = 4

epochs_amount = 100
lr = 1e-4
batch_size = 32
desired_image_size = 448

pretrained = False
pretrained_path = "some/path/"

run_description = "{}_lr={}_bs={}_grid_size={}_class_amount={}_bbox_pred={}".format(
    curr_date,
    str(lr),
    str(batch_size),
    str(grid_size),
    str(class_amount),
    str(bbox_pred_amount),
)

base_dir = os.path.join(".", "data")
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")
# train_csv_path = os.path.join(base_dir, "train.csv")
train_csv_path = os.path.join(base_dir, "100examples.csv")
# val_csv_path = os.path.join(base_dir, "val.csv")
val_csv_path = os.path.join(base_dir, "8examples.csv")

checkpoints_dir = os.path.join(base_dir, "checkpoints")
run_checkpoints_dir = os.path.join(checkpoints_dir, run_description)
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

runs_dir = os.path.join(base_dir, "runs")
curr_run_dir = os.path.join(runs_dir, run_description)
