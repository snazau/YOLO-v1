from typing import Any

import torch


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.activation = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class YOLOv1(torch.nn.Module):
    def __init__(self, class_amount, bbox_pred_amount, in_channels=3):
        super(YOLOv1, self).__init__()

        self.class_amount = class_amount
        self.grid_size = 7
        self.bbox_pred_amount = bbox_pred_amount
        self.in_channels = in_channels

        self.backbone = self.get_backbone(self.in_channels)
        self.head = self.get_head(self.grid_size, self.bbox_pred_amount, self.class_amount)

    def forward(self, x):
        x = self.backbone(x)  # [B x 1024 x grid_size x grid_size]
        x = self.head(x)
        return x

    def get_backbone(self, in_channels):
        layers = list()

        layers.append(ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3))
        layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(ConvBlock(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1))
        layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(ConvBlock(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0))
        layers.append(ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        layers.append(ConvBlock(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0))
        layers.append(ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
        layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

        for i in range(4):
            layers.append(ConvBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0))
            layers.append(ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))

        layers.append(ConvBlock(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0))
        layers.append(ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1))
        layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

        for i in range(2):
            layers.append(ConvBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0))
            layers.append(ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1))

        layers.append(ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1))
        layers.append(ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1))
        layers.append(ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1))
        layers.append(ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1))

        backbone = torch.nn.Sequential(*layers)
        return backbone

    def get_head(self, grid_size, bbox_pred_amount, class_amount):
        head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(1024 * grid_size * grid_size, 512),  # orig 4096 output neurons
            torch.nn.Dropout(0.0),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(512, grid_size * grid_size * (class_amount + 5 * bbox_pred_amount)),
        )
        return head


if __name__ == "__main__":
    import __main__
    print("Run of", __main__.__file__)
    print()

    device = torch.device("cuda:0")
    model = YOLOv1(class_amount=20, bbox_pred_amount=2, in_channels=3)
    model = model.to(device=device)
    model.train()

    input = torch.randn((32, 3, 448, 448))
    input = input.to(device)
    output = model(input)
    print("output", output.shape)

    parameters_amount = 0
    for parameter in model.parameters():
        # print(parameter.shape)
        value = 1
        for num in parameter.shape:
            value *= num

        parameters_amount += value
    print()

    print("parameters_amount:", parameters_amount)
    print("size in Gb", parameters_amount * 32 / 8 / 1024 / 1024 / 1024)
