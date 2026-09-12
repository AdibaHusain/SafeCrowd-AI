import torch
import torch.nn as nn
from torchvision import models


class CSRNet(nn.Module):
    """
    Standard CSRNet: first 10 conv layers of VGG16 as the front-end
    (pretrained on ImageNet), followed by a dilated-convolution back-end
    that expands the receptive field without further downsampling —
    important for preserving spatial resolution in the density map.
    """

    def __init__(self, load_pretrained_vgg: bool = True):
        super().__init__()

        # Front-end: matches VGG16 layers up through conv4_3
        self.frontend_layers = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
        self.frontend = self._make_layers(self.frontend_layers)

        # Back-end: dilated convs, no more max-pooling (keeps spatial size)
        self.backend_layers = [512, 512, 512, 256, 128, 64]
        self.backend = self._make_layers(self.backend_layers, in_channels=512, dilation=2)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if load_pretrained_vgg:
            self._load_pretrained_frontend()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _make_layers(self, cfg, in_channels=3, dilation=1):
        layers = []
        d_rate = dilation
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                layers += [conv, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _load_pretrained_frontend(self):
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        vgg_state = list(vgg.features.state_dict().items())
        own_state = self.frontend.state_dict()

        for (own_key, _), (_, vgg_val) in zip(own_state.items(), vgg_state):
            if own_state[own_key].shape == vgg_val.shape:
                own_state[own_key] = vgg_val
        self.frontend.load_state_dict(own_state)