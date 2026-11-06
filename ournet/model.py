import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from .convnext import ConvNeXt
nonlinearity = partial(F.relu, inplace=True)

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class Dblock_more_dilate(nn.Module):

    def __init__(self, channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(1024 + 256, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        out = x[:, :1024, :] + dilate1_out + dilate2_out + dilate3_out
        return out

class ConvNeXt_OccHeadv6(nn.Module):

    def __init__(self, num_classes=3, backbone_weights=None):
        super(ConvNeXt_OccHeadv6, self).__init__()
        self.num_classes = num_classes
        filters = [128, 256, 512, 1024]
        self.backboon = ConvNeXt()
        if backbone_weights is not None:
            state = torch.load(backbone_weights, map_location='cpu', weights_only=True)
            self.backboon.load_state_dict(state.get('model', state), strict=False)
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.dfdecoder4 = DecoderBlock(filters[3], filters[2])
        self.dfdecoder3 = DecoderBlock(filters[2], filters[1])
        self.dfdecoder2 = DecoderBlock(filters[1], filters[0])
        self.dfdecoder1 = DecoderBlock(filters[0], filters[0])
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.roadocc_head = nn.Conv2d(32, self.num_classes, kernel_size=2, padding=1)
        self.wofinaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.wofinalrelu1 = nonlinearity
        self.wofinalconv2 = nn.Conv2d(32, 32, 3)
        self.wofinalrelu2 = nonlinearity
        self.roadwo_head = nn.Conv2d(32, 2, kernel_size=2, padding=1)
        self.reduce_dim_0 = nn.Sequential(nn.Conv2d(1280, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.d_module = Dblock_more_dilate(1024)

    def forward(self, x_dict):
        x = x_dict['img']
        if x.shape[1:] != (3, 512, 512):
            raise ValueError('Expected RGB images of shape [B, 3, 512, 512]')
        img_sdpt = x_dict['img_sdpt']
        '\n        img_sdpt shape\n        [0]:1280,16,16\n        [1]:1280,32,32\n        [2]:640,64,64\n        [3]:320,64,64\n        [4]:320,32,32\n        '
        x4 = self.backboon(x)
        e4 = x4[3]
        e3 = x4[2]
        e2 = x4[1]
        e1 = x4[0]
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        sd0 = self.reduce_dim_0(img_sdpt[0])
        dfd4 = self.d_module(torch.cat((e4, sd0), dim=1))
        dfd4 = self.dfdecoder4(dfd4) + e3
        dfd3 = self.dfdecoder3(dfd4) + e2
        dfd2 = self.dfdecoder2(dfd3) + e1
        dfd1 = self.dfdecoder1(dfd2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        roadocc_masks = self.roadocc_head(out)
        woout = self.wofinaldeconv1(dfd1)
        woout = self.wofinalrelu1(woout)
        woout = self.wofinalconv2(woout)
        woout = self.wofinalrelu2(woout)
        roadwo_masks = self.roadwo_head(woout)
        B = roadwo_masks.shape[0]
        return {'roadocc_masks': roadocc_masks, 'roadwo_masks': torch.sigmoid(roadwo_masks[:, 0].reshape(B, 1, 512, 512)), 'roadwo_edge_masks': torch.sigmoid(roadwo_masks[:, 1].reshape(B, 1, 512, 512))}
