import torch.nn as nn
from torchvision import models


def _build_resnet18(num_classes: int):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_mobilenet_v2(num_classes: int):
    model = models.mobilenet_v2(weights=None)
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        3,
        first_conv.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_model(name: str = "resnet18", num_classes: int = 2):
    builders = {
        "resnet18": _build_resnet18,
        "mobilenet_v2": _build_mobilenet_v2,
    }
    if name not in builders:
        raise ValueError(f"Unsupported model '{name}'. Available: {list(builders)}")
    return builders[name](num_classes=num_classes)
