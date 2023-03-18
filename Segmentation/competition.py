import torch
import torch.nn as nn
import torch.nn.functional as F


class DC(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if dropout:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.LeakyReLU(inplace=True),
                nn.Dropout2d(),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
            )

    def forward(self, x):
        return self.double_conv(x)


class DownsampleGroup(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DC(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpsampleGroup(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, dropout=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DC(in_channels, out_channels, in_channels // 2, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DC(in_channels, out_channels, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class model(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(model, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DC(n_channels, 64))
        self.down1 = (DownsampleGroup(64, 128))
        self.down2 = (DownsampleGroup(128, 256))
        self.down3 = (DownsampleGroup(256, 512))

        factor = 2 if bilinear else 1

        self.down4 = (DownsampleGroup(512, 1024 // factor, dropout=True))

        self.up1 = (UpsampleGroup(1024, 512 // factor, bilinear))
        self.up2 = (UpsampleGroup(512, 256 // factor, bilinear, dropout=True))
        self.up3 = (UpsampleGroup(256, 128 // factor, bilinear))
        self.up4 = (UpsampleGroup(128, 64, bilinear))

        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)

        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)

        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)

        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)

        self.outc = torch.utils.checkpoint(self.outc)

    def save_model(self, path="model.pth"):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        model.load_state_dict(torch.load(path))
        return model