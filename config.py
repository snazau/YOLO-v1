import datetime
import os
import torch

curr_date = datetime.datetime.now().strftime("%b%d_%H-%M-%S")

grid_size = 7
class_amount = 20
bbox_pred_amount = 3

debug = False
print_freq = 1000
device = torch.device("cuda:0")
loader_workers = 4

epochs_amount = 100
lr = 2e-5
batch_size = 32
desired_image_size = 448

confidence_threshold = 0.7
iou_threshold = 0.5

pretrained = False
pretrained_path = "some/path/"

run_description = "{}_lr={}_bs={}_grid_size={}_class_amount={}_bbox_pred={}_ds={}".format(
    curr_date,
    str(lr),
    str(batch_size),
    str(grid_size),
    str(class_amount),
    str(bbox_pred_amount),
    "full",
)

base_dir = os.path.join(".", "data")
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")
train_csv_path = os.path.join(base_dir, "train.csv")
val_csv_path = os.path.join(base_dir, "val.csv")

checkpoints_dir = os.path.join(base_dir, "checkpoints")
run_checkpoints_dir = os.path.join(checkpoints_dir, run_description)
if not os.path.exists(run_checkpoints_dir):
    os.makedirs(run_checkpoints_dir)

runs_dir = os.path.join(base_dir, "runs")
curr_run_dir = os.path.join(runs_dir, run_description)
