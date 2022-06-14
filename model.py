import torch
from torch import nn
from torchvision.models import resnet18


class Shift_model (nn.Module) :
    def __init__(self):
        super(Shift_model, self).__init__()
        self.encoder = self._get_correct_resnet()
        self.out = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 3, kernel_size=1, padding=0),
        )

    def forward(self, x, training=True):
        heatmap = self.encoder(x)

        if training:
            heatmap = nn.AdaptiveAvgPool2d(1)(heatmap)
            out = self.out(heatmap)
            out[:, 0] = nn.Sigmoid()(out[:, 0])
            out = out.view(-1, 3)
        else:
            out = self.out(heatmap)
            out[:, 0] = nn.Sigmoid()(out[:, 0])

        return out

    def _get_correct_resnet(self):
        resnet = resnet18(pretrained=True)
        resnet_encoder_list = list(resnet.children())[:-2]
        resnet = nn.Sequential(*resnet_encoder_list)
        return resnet
