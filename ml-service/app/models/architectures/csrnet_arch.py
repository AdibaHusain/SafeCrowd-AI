import torch
import torch.nn as nn
from torchvision import models


class CSRNet(nn.Module):
    """
    CSRNet architecture used by the SafeCrowdAI inference service.

    Frontend:
        VGG16-based convolutional layers with 3 pooling stages.

    Backend:
        Dilated convolutional layers that preserve spatial resolution.

    Output:
        1-channel density map.
    """

    def __init__(self, load_pretrained_vgg: bool = False):
        super().__init__()

        # VGG16 frontend up to conv4_3
        self.frontend_layers = [
            64, 64, "M",
            128, 128, "M",
            256, 256, 256, "M",
            512, 512, 512
        ]

        self.frontend = self._make_layers(self.frontend_layers)

        # Dilated backend
        self.backend_layers = [
            512, 512, 512,
            256, 128, 64
        ]

        self.backend = self._make_layers(
            self.backend_layers,
            in_channels=512,
            dilation=2
        )

        # Final density-map prediction layer
        self.output_layer = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

        # Only needed when initializing a new model for training.
        # During inference, the trained checkpoint already contains
        # the required weights.
        if load_pretrained_vgg:
            self._load_pretrained_frontend()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _make_layers(
        self,
        cfg,
        in_channels=3,
        dilation=1
    ):
        layers = []
        d_rate = dilation

        for v in cfg:
            if v == "M":
                layers.append(
                    nn.MaxPool2d(
                        kernel_size=2,
                        stride=2
                    )
                )
            else:
                conv = nn.Conv2d(
                    in_channels,
                    v,
                    kernel_size=3,
                    padding=d_rate,
                    dilation=d_rate
                )

                layers.extend([
                    conv,
                    nn.ReLU(inplace=True)
                ])

                in_channels = v

        return nn.Sequential(*layers)

    def _load_pretrained_frontend(self):
        vgg = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
        )

        vgg_state = list(
            vgg.features.state_dict().items()
        )

        own_state = self.frontend.state_dict()

        for (own_key, _), (_, vgg_val) in zip(
            own_state.items(),
            vgg_state
        ):
            if own_state[own_key].shape == vgg_val.shape:
                own_state[own_key] = vgg_val

        self.frontend.load_state_dict(own_state)