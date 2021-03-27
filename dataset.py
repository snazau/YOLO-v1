import cv2
import numpy as np
import os
import pandas as pd
import torch

import config
import utils


class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir, csv_path, image_size, grid_size, class_amount, transform=None):
        self.images_dir = images_dir
        self.filenames = []
        self.labels = []

        self.images_size = image_size

        self.grid_size = grid_size
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
        image_path = os.path.join(self.images_dir, self.filenames[index])
        image_bgr = cv2.imread(image_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        new_shape = (self.images_size, self.images_size)
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


def visualize_sample(sample, grid_size, class_amount):
    import copy
    image = copy.deepcopy(sample["tensor"])
    image = utils.chw2hwc(image)
    image = image.detach().cpu().numpy()
    image *= 255
    image = image.astype(np.uint8)

    image_height, image_width = image.shape[:2]

    label = sample["label"]
    objects_amount = 0
    for cell_i in range(grid_size):
        for cell_j in range(grid_size):
            cell_info = label[cell_i, cell_j]
            is_obj_presented = cell_info[class_amount]
            if is_obj_presented == 1:
                objects_amount += 1

                cell_relative_center_x = cell_info[class_amount + 1]
                cell_relative_center_y = cell_info[class_amount + 2]
                width_in_cell_size = cell_info[class_amount + 3]
                height_in_cell_size = cell_info[class_amount + 4]

                image_relative_cell_size = 1 / grid_size
                image_relative_center_x = cell_j * image_relative_cell_size + cell_relative_center_x * image_relative_cell_size
                image_relative_center_y = cell_i * image_relative_cell_size + cell_relative_center_y * image_relative_cell_size
                width_in_image_size = width_in_cell_size / grid_size
                height_in_image_size = height_in_cell_size / grid_size

                color = [np.random.randint(0, 255) for _ in range(3)]
                bbox_title = str(objects_amount) + "_o"
                object_bbox = [
                    (image_relative_center_x - width_in_image_size / 2) * image_width,
                    (image_relative_center_y - height_in_image_size / 2) * image_height,
                    (image_relative_center_x + width_in_image_size / 2) * image_width,
                    (image_relative_center_y + height_in_image_size / 2) * image_height,
                ]
                utils.plot_one_box(object_bbox, image, color, bbox_title)

                cell_bbox = [
                    image_relative_cell_size * cell_j * image_width,
                    image_relative_cell_size * cell_i * image_height,
                    image_relative_cell_size * (cell_j + 1) * image_width,
                    image_relative_cell_size * (cell_i + 1) * image_height,
                ]
                color = [np.random.randint(0, 255) for _ in range(3)]
                bbox_title = str(objects_amount) + "_c"
                utils.plot_one_box(cell_bbox, image, color, bbox_title)

    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    import __main__
    print("Run of", __main__.__file__)

    # Reproducibility
    # torch.manual_seed(0)
    #
    import random
    # random.seed(0)
    #
    # import numpy as np
    # np.random.seed(0)

    # Data
    base_dir = os.path.join(".", "data")
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    csv_path = os.path.join(base_dir, "8examples.csv")

    grid_size = 7
    class_amount = 20

    dataset = YOLODataset(images_dir, labels_dir, csv_path, config.desired_image_size, grid_size, class_amount)
    random_index = random.randint(0, len(dataset) - 1)
    # random_index = 0
    sample = dataset[random_index]
    visualize_sample(sample, grid_size, class_amount)

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
