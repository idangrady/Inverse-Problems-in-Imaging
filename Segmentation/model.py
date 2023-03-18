import torch
import torch.nn as nn
import torchvision.transforms.functional as TF



class Doubleconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Doubleconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels =3,
                 out_channels=1,
                 features= [64, 128,256,512],
                 ):
        super(UNET, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #create downpart for every group -> three every tume
        for feature in features:
            self.downs.append(Doubleconv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(Doubleconv(feature*2, feature))


        self.bottleneck = Doubleconv(features[-1], features[-1]*2,) #1024
        self.finalConv = nn.Conv2d(features[0], out_channels,kernel_size=1)


    def forward(self, x):
        skip_connect =[]

        for down in self.downs:
            x = down(x)
            skip_connect.append(x)
            x = self.pool(x)

        # arrived to the bottelneck

        x = self.bottleneck(x)
        skip_connect = skip_connect[: : -1]


        for idx in range(0,len(self.ups),2):
            x = self.ups[idx](x)
            concat_layer =skip_connect[idx//2]
            if x.shape != concat_layer.shape:
                x = TF.resize(x, size=concat_layer.shape[2:])
            concat = torch.cat((x, concat_layer), dim=1)# concate across the first dimention

            # now we should use the second Doubleconv
            x = self.ups[idx+1](concat)

        return self.finalConv(x)


def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()
    print("Done")