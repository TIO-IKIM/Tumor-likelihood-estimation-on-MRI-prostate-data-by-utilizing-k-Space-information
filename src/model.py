import timm
import torch.nn as nn


class ConvNext(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super(ConvNext, self).__init__()

        self.encoder = timm.create_model(
            "convnext_atto",
            pretrained=True,
            in_chans=in_channels,
            num_classes=num_classes,
        )

    def forward(self, x):

        return self.encoder(x)
