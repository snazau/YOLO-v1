import cv2
import os
import pandas as pd
import torch

import utils


class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir, csv_path, grid_size, bbox_pred_amount, class_amount, transform=None):
        self.images_dir = images_dir
        self.filenames = []
        self.labels = []

        self.grid_size = grid_size
        self.bbox_pred_amount = bbox_pred_amount
        self.class_amount = class_amount

        df = pd.read_csv(csv_path)
        for index, row in df.iterrows():
            label_path = os.path.join(labels_dir, row["label"])
            label = YOLODataset._parse_label(label_path)
            self.labels.append(label)
            self.filenames.append(row["image"])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image_path = os.path.join(images_dir, self.filenames[index])
        image_bgr = cv2.imread(image_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        new_shape = (448, 448)
        image = cv2.resize(image, new_shape, interpolation=cv2.INTER_CUBIC)
        image = image.astype(np.float32)
        image /= 255.0
        tensor = utils.hwc2chw(image)
        tensor = torch.from_numpy(tensor)

        label_tensor = torch.zeros((self.grid_size, self.grid_size, self.class_amount + 5))
        for bbox in self.labels[index]:
            class_id = bbox["class_id"]

            cell_i = int(self.grid_size * bbox["center_y"])
            cell_j = int(self.grid_size * bbox["center_x"])

            # Relative coordinates ([0, 1]) where 1 = cell_size
            center_in_cell_size_x = bbox["center_x"] * self.grid_size - cell_j
            center_in_cell_size_y = bbox["center_y"] * self.grid_size - cell_i

            # Relative coordinates ([0, 1]) where 1 = cell_size
            width_in_cell_size = bbox["width"] * self.grid_size
            height_in_cell_size = bbox["height"] * self.grid_size

            if label_tensor[cell_i, cell_j, self.class_amount] == 0:  # since in specific cell can be only 1 object
                label_tensor[cell_i, cell_j, class_id] = 1  # setting 1 in one-hot label
                label_tensor[cell_i, cell_j, self.class_amount] = 1  # obj presented = 1
                label_tensor[cell_i, cell_j, self.class_amount + 1] = center_in_cell_size_x
                label_tensor[cell_i, cell_j, self.class_amount + 2] = center_in_cell_size_y
                label_tensor[cell_i, cell_j, self.class_amount + 3] = width_in_cell_size
                label_tensor[cell_i, cell_j, self.class_amount + 4] = height_in_cell_size

        sample = {
            "tensor": tensor,
            "label": label_tensor,
            "filename": self.filenames[index],
        }

        return sample

    @staticmethod
    def _parse_label(label_path):
        bboxes = []
        with open(label_path) as file:
            for line in file.readlines():
                line_values = line.split(" ")
                assert len(line_values) == 5, "expected that 5 values describes the bbox but find {} values instead".format(len(line_values))

                bbox = {
                    "class_id": int(line_values[0]),
                    "center_x": float(line_values[1]),
                    "center_y": float(line_values[2]),
                    "width": float(line_values[3]),
                    "height": float(line_values[4]),
                }
                bboxes.append(bbox)
        return bboxes


if __name__ == "__main__":
    import __main__
    print("Run of", __main__.__file__)

    # Reproducibility
    torch.manual_seed(0)

    import random
    random.seed(0)

    import numpy as np
    np.random.seed(0)

    # Data
    base_dir = os.path.join(".", "data")
    images_dir = os.path.join(base_dir, "one_image")
    labels_dir = os.path.join(base_dir, "one_label")
    csv_path = os.path.join(base_dir, "one_sample.csv")

    grid_size = 7
    bbox_pred_amount = 2
    class_amount = 20

    dataset = YOLODataset(images_dir, labels_dir, csv_path, grid_size, bbox_pred_amount, class_amount)
    sample = dataset[0]
    tensor = sample["tensor"]
    tensor_label = sample["label"]
    filename = sample["filename"]

    print("filename", filename)
    print("tensor", tensor.shape, type(tensor), tensor.dtype, tensor.min(), tensor.max())
    print("tensor_label", tensor_label.shape, type(tensor_label), tensor_label.dtype, tensor_label.min(), tensor_label.max())
    print()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for index, sample in enumerate(dataloader):
        print("#" + str(index), sample["filename"])

        inputs = sample["tensor"]
        labels = sample["label"]
        filenames = sample["filename"]

        print("inputs", inputs.shape, type(inputs), inputs.dtype, inputs.min(), inputs.max())
        print("labels", labels.shape, type(labels), labels.dtype, labels.min(), labels.max())
