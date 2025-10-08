# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models

class SaliencyNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None).features
        # Encoder stages (downsample by MaxPool)
        self.enc1 = vgg[0:6]    # 64
        self.enc2 = vgg[6:13]   # 128
        self.enc3 = vgg[13:23]  # 256
        self.enc4 = vgg[23:33]  # 512
        self.enc5 = vgg[33:43]  # 512

        def up(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
        self.up5 = up(512, 512)
        self.up4 = up(512, 256)
        self.up3 = up(256, 128)
        self.up2 = up(128, 64)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.enc1(x)              # H/1
        x2 = self.enc2(x1)             # H/2
        x3 = self.enc3(x2)             # H/4
        x4 = self.enc4(x3)             # H/8
        x5 = self.enc5(x4)             # H/16
        d5 = self.up5(x5)              # H/8
        d4 = self.up4(d5)              # H/4
        d3 = self.up3(d4)              # H/2
        d2 = self.up2(d3)              # H
        out = self.up1(d2)             # 1xHxW (logits)
        return out