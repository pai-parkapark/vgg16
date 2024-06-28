import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_model import BaseModel


class VGGBlock(BaseModel):
    def __init__(self, in_channels, out_channels, num_convs):
        super(VGGBlock, self).__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGG16(BaseModel):
    def __init__(self, num_classes=100):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            VGGBlock(3, 64, 2),  # Two 3x3 conv layers, 64 filters
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGGBlock(64, 128, 2),  # Two 3x3 conv layers, 128 filters
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGGBlock(128, 256, 3),  # Three 3x3 conv layers, 256 filters
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGGBlock(256, 512, 3),  # Three 3x3 conv layers, 512 filters
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGGBlock(512, 512, 3),  # Three 3x3 conv layers, 512 filters
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    num_classes = 10
    model = VGG16(num_classes=num_classes)
    print(model)
