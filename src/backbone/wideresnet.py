import torch
import torch.nn as nn
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights


class WideResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        weights = Wide_ResNet50_2_Weights.DEFAULT
        model = wide_resnet50_2(weights=weights)

        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

        self.eval()

        for parameter in self.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)

        features1 = self.layer2(x)
        features2 = self.layer3(features1)

        return [features1, features2]
