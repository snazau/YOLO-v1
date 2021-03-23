import torch

import config_architecture


def create_conv_from_description(layer_description, in_channels):
    layer = CNNBlock(
        in_channels,
        layer_description["params"]["output_filter_amount"],
        kernel_size=layer_description["params"]["kernel_size"],
        stride=layer_description["params"]["stride"],
        padding=layer_description["params"]["padding"],
    )
    return layer


def create_maxpool_from_description(layer_description):
    layer = torch.nn.MaxPool2d(
        layer_description["params"]["kernel_size"],
        stride=layer_description["params"]["stride"],
    )
    return layer


def create_layer_from_description(layer_description, in_channels):
    if layer_description["type"] == "conv":
        layer = create_conv_from_description(layer_description, in_channels)
        return layer
    elif layer_description["type"] == "maxpool":
        layer = create_maxpool_from_description(layer_description)
        return layer
    else:
        raise NotImplemented()


class CNNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNNBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.activation = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class YOLOv1(torch.nn.Module):
    def __init__(self, backbone_config, class_amount, bbox_pred_amount, in_channels=3):
        super(YOLOv1, self).__init__()
        self.backbone_config = backbone_config

        self.class_amount = class_amount
        self.grid_size = 7
        self.bbox_pred_amount = bbox_pred_amount
        self.in_channels = in_channels

        self.backbone = self.get_backbone(backbone_config)
        self.head = self.get_head(self.grid_size, self.bbox_pred_amount, self.class_amount)

    def forward(self, x):
        x = self.backbone(x)  # [B x 1024 x grid_size x grid_size]
        x = self.head(x)
        return x

    def get_backbone(self, backbone_config):
        layers = []
        in_channels = self.in_channels

        for layer_description in backbone_config:
            if layer_description["type"] == "conv":
                layer = create_layer_from_description(layer_description, in_channels)
                layers.append(layer)
                in_channels = layer_description["params"]["output_filter_amount"]
            elif layer_description["type"] == "maxpool":
                layer = create_layer_from_description(layer_description, in_channels)
                layers.append(layer)
            elif layer_description["type"] == "repeated_block":
                for repeats_amount in range(layer_description["repeats_amount"]):
                    for nested_layer_description in layer_description["layers_descriptions"]:
                        layer = create_layer_from_description(nested_layer_description, in_channels)
                        in_channels = nested_layer_description["params"]["output_filter_amount"]
                        layers.append(layer)

        # for layer_index, layer in enumerate(layers):
        #     print("layer #" + str(layer_index), layer)

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
    model = YOLOv1(config_architecture.backbone_config, class_amount=20, bbox_pred_amount=2, in_channels=3)
    model = model.to(device=device)

    input = torch.randn((32, 3, 448, 448))
    input = input.to(device)
    output = model(input)
    print("output", output.shape)

    parameters_amount = 0
    for parameter in model.parameters():
        print(parameter.shape)

        value = 1
        for num in parameter.shape:
            value *= num

        parameters_amount += value
    print()

    print("parameters_amount:", parameters_amount)
    print("size in Gb", parameters_amount * 32 / 8 / 1024 / 1024 / 1024)