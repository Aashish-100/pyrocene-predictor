import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class FuelCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # ❶ Load Imagenet-trained MobileNet-V2 backbone
        mnet = mobilenet_v2(weights="MobileNet_V2_Weights.IMAGENET1K_V1")  # torchvision ≥ 0.15
        self.features = mnet.features          # all conv layers
        for p in self.features.parameters():
            p.requires_grad = False            # freeze backbone

        # ❷ Replace first conv to accept 4 channels
        old = self.features[0][0]              # Conv2d(3,32,3,stride=2)
        new = nn.Conv2d(4, 32, 3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight     # copy RGB weights
            new.weight[:, 3]  = old.weight[:, 0]  # init 4th band ≈ red
        self.features[0][0] = new

        # ❸ Replace classifier head
        self.avgpool   = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x).squeeze(1)   # (N,)