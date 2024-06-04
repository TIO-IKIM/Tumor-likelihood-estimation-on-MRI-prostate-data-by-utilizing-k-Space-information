import timm
import torch.nn as nn


class ConvNext(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, split: bool) -> None:
        super(ConvNext, self).__init__()

        self.split = split

        if split:
            self.encoder_i = timm.create_model(
            "convnext_atto",
            pretrained=True,
            in_chans=1,
            num_classes=num_classes,
        )

            self.encoder_k = timm.create_model(
            "convnext_atto",
            pretrained=True,
            in_chans=2,
            num_classes=num_classes,
        )
        else:
            self.encoder = timm.create_model(
            "convnext_atto",
            pretrained=True,
            in_chans=in_channels,
            num_classes=num_classes,
        )

    def forward(self, x):

        if self.split:
            x_i = self.encoder_i(x[:, :2,...].float())
            x_k = self.encoder_k(x[:, 2:, ...].float())

            return (x_i + x_k) / 2
        
        else:
            return self.encoder(x)
