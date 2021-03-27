import os
import torch

grid_size = 7
class_amount = 20
bbox_pred_amount = 2

debug = True
print_freq = 1000
device = torch.device("cuda:0")

epochs_amount = 100
lr = 1e-4
batch_size = 32
desired_image_size = 448

pretrained = False
pretrained_path = "some/path/"

base_dir = os.path.join(".", "data")
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")
train_csv_path = os.path.join(base_dir, "train.csv")
val_csv_path = os.path.join(base_dir, "val.csv")
