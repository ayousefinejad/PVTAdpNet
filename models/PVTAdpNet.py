import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# Assuming pvt_v2_b2 is correctly implemented and available
from src.pvtv2 import pvt_v2_b2

class AdapterLayer(nn.Module):
    def __init__(self, in_c, out_c, bottleneck, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, activation='relu'):
        super(AdapterLayer, self).__init__()

        self.up_project = nn.Conv2d(in_c, bottleneck, kernel_size=1, padding=0, bias=bias)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.down_project = nn.Conv2d(bottleneck, out_c, kernel_size=1, padding=0, bias=bias)
        self.norm = nn.BatchNorm2d(out_c)

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


class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=True, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x

class AdapterLayer(nn.Module):
    def __init__(self, in_c, out_c, bottleneck, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, activation='relu'):
        super(AdapterLayer, self).__init__()

        self.up_project = nn.Conv2d(in_c, bottleneck, kernel_size=1, padding=0, bias=bias)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.down_project = nn.Conv2d(bottleneck, out_c, kernel_size=1, padding=0, bias=bias)
        self.norm = nn.BatchNorm2d(out_c)

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

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, bottleneck):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = AdapterLayer(in_c + out_c, out_c, bottleneck)

    def forward(self, x, s):
        x = self.up(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, scale, bottleneck):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.r1 = AdapterLayer(in_c, out_c, bottleneck)

    def forward(self, inputs):
        x = self.up(inputs)
        x = self.r1(x)
        return x

class PvtAdapNet(nn.Module):
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

        """ Channel Reduction """
        self.c1 = Conv2D(64, 64, kernel_size=1, padding=0)
        self.c2 = Conv2D(128, 64, kernel_size=1, padding=0)
        self.c3 = Conv2D(320, 64, kernel_size=1, padding=0)

        self.d1 = DecoderBlock(64, 64, bottleneck=32)
        self.d2 = DecoderBlock(64, 64, bottleneck=32)
        self.d3 = UpBlock(64, 64, 4, bottleneck=32)

        self.u1 = UpBlock(64, 64, 4, bottleneck=32)
        self.u2 = UpBlock(64, 64, 8, bottleneck=32)
        self.u3 = UpBlock(64, 64, 16, bottleneck=32)

        self.r1 = AdapterLayer(64*4, 64, bottleneck=64)
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
    model = PvtAdapNet()
    y = model(x)
    print(y.shape)
