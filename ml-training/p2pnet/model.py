import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn, VGG16_BN_Weights

from anchor_points import generate_anchor_points, shift_anchor_points


class VGGBackbone(nn.Module):
    """
    Extracts two feature levels from VGG16_bn: stride-8 (shallower, more
    spatial detail — good for small/distant heads) and stride-16 (deeper,
    more semantic). Layer indices below are the VGG16_bn stage boundaries —
    verify with the shape-test if you change torchvision versions.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = vgg16_bn(weights=weights).features
        layers = list(vgg.children())
        self.stage_to_stride8 = nn.Sequential(*layers[:33])     # input -> stride 8
        self.stage_to_stride16 = nn.Sequential(*layers[33:43])  # stride 8 -> stride 16

    def forward(self, x):
        feat8 = self.stage_to_stride8(x)
        feat16 = self.stage_to_stride16(feat8)
        return feat8, feat16


class RegressionHead(nn.Module):
    """Predicts a (dx, dy) pixel offset for every anchor point."""
    def __init__(self, in_channels=256, num_anchor_points=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.output = nn.Conv2d(256, num_anchor_points * 2, 3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.output(x)
        b, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, -1, 2)
        return x.reshape(b, -1, 2)


class ClassificationHead(nn.Module):
    """Predicts foreground(person)/background logits for every anchor point."""
    def __init__(self, in_channels=256, num_anchor_points=4, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.output = nn.Conv2d(256, num_anchor_points * num_classes, 3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.output(x)
        b, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, -1, self.num_classes)
        return x.reshape(b, -1, self.num_classes)


class P2PNet(nn.Module):
    """
    Simplified P2PNet: VGG16_bn backbone -> lightweight top-down feature
    merge (stride-16 upsampled into stride-8) -> per-anchor-point regression
    (xy offset) + classification (person/background) heads.

    WORKING SIMPLIFICATION — compare against the team's actual Colab
    notebook before treating results as final (same correction pass CSRNet
    needed in Step 1).
    """
    def __init__(self, row=2, line=2, pretrained=True):
        super().__init__()
        self.stride = 8
        self.anchor_points = generate_anchor_points(self.stride, row, line)
        num_anchor_points = row * line

        self.backbone = VGGBackbone(pretrained=pretrained)
        self.lateral16 = nn.Conv2d(512, 256, 1)
        self.lateral8 = nn.Conv2d(256, 256, 1)

        self.reg_head = RegressionHead(256, num_anchor_points)
        self.cls_head = ClassificationHead(256, num_anchor_points, num_classes=2)

    def forward(self, x):
        feat8, feat16 = self.backbone(x)

        p16 = self.lateral16(feat16)
        p16_up = F.interpolate(p16, size=feat8.shape[-2:], mode="nearest")
        p8 = self.lateral8(feat8) + p16_up

        reg_offsets = self.reg_head(p8)   # (B, N, 2) — pixel offsets
        cls_logits = self.cls_head(p8)    # (B, N, 2) — fg/bg logits

        anchors = shift_anchor_points(p8.shape[-2:], self.stride, self.anchor_points)
        anchors = torch.tensor(anchors, dtype=torch.float32, device=x.device)
        pred_points = anchors.unsqueeze(0) + reg_offsets

        return pred_points, cls_logits