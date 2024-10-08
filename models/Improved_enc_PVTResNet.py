import torch
import torch.nn as nn
from src.pvtv2 import pvt_v2_b2

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

class improved_enc_PVTResNet(nn.Module):
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

        """ Channel Matching for Summation """
        self.match1 = nn.Conv2d(64, 128, kernel_size=1, padding=0)
        self.match2 = nn.Conv2d(128, 320, kernel_size=1, padding=0)

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
        backbone_outputs = self.backbone(inputs)
        e1 = backbone_outputs[0]  ## [-1, 64, h/4, w/4]
        e2 = backbone_outputs[1]  ## [-1, 128, h/8, w/8]
        e3 = backbone_outputs[2]  ## [-1, 320, h/16, w/16]

        # Downsample e1 to match the spatial dimensions of e2 and match channels
        e1_downsampled = nn.functional.adaptive_avg_pool2d(e1, output_size=e2.shape[2:])
        e1_downsampled = self.match1(e1_downsampled)
        e1_e2_sum = e1_downsampled + e2

        # Downsample e2 to match the spatial dimensions of e3 and match channels
        e2_downsampled = nn.functional.adaptive_avg_pool2d(e2, output_size=e3.shape[2:])
        e2_downsampled = self.match2(e2_downsampled)
        e2_e3_sum = e2_downsampled + e3

        c1 = self.c1(e1)
        c2 = self.c2(e1_e2_sum)
        c3 = self.c3(e2_e3_sum)

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
    model = improved_enc_PVTResNet()
    y = model(x)
    print(y.shape)