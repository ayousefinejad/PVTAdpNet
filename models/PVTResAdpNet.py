import torch
import torch.nn as nn
from src.pvtv2 import pvt_v2_b2

class AdapterLayer(nn.Module):
    def __init__(self, in_c, out_c, bottleneck, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, activation='relu'):
        super(AdapterLayer, self).__init__()

        self.up_project = nn.Conv2d(in_c, bottleneck, kernel_size=1, padding=0, bias=bias)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.down_project = nn.Conv2d(bottleneck, out_c, kernel_size=1, padding=0, bias=bias)
        self.norm = nn.BatchNorm2d(out_c)

        # Ensure the stride and padding are set to match the Conv2D layer
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0, bias=bias),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x):
        residual = x
        out = self.up_project(x)
        out = self.activation(out)
        out = self.down_project(out)
        out = self.norm(out)
        out += self.shortcut(residual)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, inputs):
        x1 = self.conv(inputs)
        x2 = self.shortcut(inputs)
        x = self.relu(x1 + x2)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c + out_c, out_c)

    def forward(self, x, s):
        x = self.up(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, scale):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c, out_c)

    def forward(self, inputs):
        x = self.up(inputs)
        x = self.r1(x)
        return x

class PvtResAdapNet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.backbone = pvt_v2_b2()  ## [64, 128, 320, 512]
        path = 'pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        """ Channel Reduction with AdapterLayer """
        self.c1 = AdapterLayer(64, 64, bottleneck=32, kernel_size=3, padding=1, stride=1, activation='relu')
        self.c2 = AdapterLayer(128, 64, bottleneck=64, kernel_size=3, padding=1, stride=1, activation='relu')
        self.c3 = AdapterLayer(320, 64, bottleneck=160, kernel_size=3, padding=1, stride=1, activation='relu')

        self.d1 = DecoderBlock(64, 64)
        self.d2 = DecoderBlock(64, 64)
        self.d3 = UpBlock(64, 64, 4)

        self.u1 = UpBlock(64, 64, 4)
        self.u2 = UpBlock(64, 64, 8)
        self.u3 = UpBlock(64, 64, 16)

        self.r1 = ResidualBlock(64 * 4, 64)
        self.y = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        pvt1 = self.backbone(inputs)
        e1 = pvt1[0]     ## [-1, 64, h/4, w/4]
        e2 = pvt1[1]     ## [-1, 128, h/8, w/8]
        e3 = pvt1[2]     ## [-1, 320, h/16, w/16]

        c1 = self.c1(e1)
        c2 = self.c2(e2)
        c3 = self.c3(e3)

        d1 = self.d1(c3, c2)
        d2 = self.d2(d1, c1)
        d3 = self.d3(d2)

        u1 = self.u1(c1)
        u2 = self.u2(c2)
        u3 = self.u3(c3)

        x = torch.cat([d3, u1, u2, u3], axis=1)
        x = self.r1(x)
        y = self.y(x)
        return y

if __name__ == "__main__":
    x = torch.randn((4, 3, 256, 256))
    model = PvtResAdapNet()
    y = model(x)
    print(y.shape)
