import torch
import torch.nn as nn
import os

from models.darknet19 import DarkNet19


class YOLOv2(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv2, self).__init__()
        darknet19 = DarkNet19(100)
        darknet19.load_state_dict(torch.load('../checkpoints/DarkNet19.pth'))
        # input shape [B, 3, 416, 416]
        self.module_1 = darknet19.module_1
        self.module_2 = darknet19.module_2
        self.module_3 = darknet19.module_3
        self.module_4 = darknet19.module_4
        self.module_5 = darknet19.module_5
        self.module_6 = darknet19.module_6
        self.module_7 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, num_classes, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.module_1(x)
        x = self.module_2(x)
        x = self.module_3(x)
        x = self.module_4(x)
        skip_x = x = self.module_5(x)
        x = self.module_6(x)
        skip_x = skip_x.reshape([-1, 2048, 13, 13])
        x = torch.cat([x, skip_x], dim=1)


if __name__ == '__main__':
    print(YOLOv2(5))