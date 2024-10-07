# The SegFormer code was heavily based on https://github.com/NVlabs/SegFormer
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/NVlabs/SegFormer#license

import imp
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils
from paddleseg.models.segformer import SegFormer_B3, SegFormer_B32
from paddleseg.models.dem_feature import Generator
from paddleseg.models.hrnet_w48 import HRNetW48Contrast
from paddleseg.models.sar_feature import Generator_sar
from paddleseg.models.ocrnet import OCRNet


# from .backbones import MixVisionTransformer_B4


class h_sigmoid(nn.Layer):
    def __init__(self):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6


class SE(nn.Layer):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2D((1, 1))
        self.compress = nn.Conv2D(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2D(in_chnls // ratio, in_chnls, 1, 1, 0)
        # self.conv = nn.Conv2D(in_chnls,in_chnls//2,1,1,0)

    def forward(self, x1, x2):
        x = paddle.concat([x1, x2], axis=1)
        # x = self.conv(x1)
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        # x1.shape: [2, 720, 256, 256]
        # out.shape: [2, 392, 1, 1]

        return x * F.sigmoid(out)


class Fusion(nn.Layer):
    def __init__(self, inp1, inp2, reduction=32):
        super(Fusion, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2D((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2D((1, None))

        self.se = SE(inp1 + inp2, 16)  #

        mip = max(8, (inp1 + inp2) // reduction)

        self.conv1 = nn.Conv2D(inp1 + inp2, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2D(mip)
        self.act = h_sigmoid()

        self.conv_h = nn.Conv2D(1024 + 720 + 1170, 1, kernel_size=(7, 1), stride=1, padding=(3, 0))

        self.conv_w = nn.Conv2D(1024 + 720 + 1170, 1, kernel_size=(1, 7), stride=1, padding=(0, 3))

        # self.conv_c = nn.Conv2D(2, 1, kernel_size=7, padding=7 // 2)

        self.gap = nn.AdaptiveAvgPool2D(1)
        self.conv1d = nn.Conv1D(1, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Sequential(
            nn.Conv2D(1024 + 720 + 1170, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Conv2D(1024, 1024, 1),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Conv2D(1024 + 720 + 1170, 720 + 1170, 3, padding=1),
            nn.ReLU(),
            nn.Conv2D(720 + 1170, 720 + 1170, 1),
            nn.Sigmoid()
        )

    def forward(self, x11, x12, x2):  # opt,dem,hill
        x1 = x11  # hillshade+opt   (1024)
        x2 = self.se(x12, x2)  # 1890(720+1170)

        identity_x1 = x1
        identity_x2 = x2  # 1890
        b1, c1, _, _ = x1.shape
        b2, c2, _, _ = x2.shape
        # 通道注意力
        x1_x2 = paddle.concat([x1, x2], axis=1)  # 1024 + 720 + 1170
        gap = self.gap(x1_x2)
        x1_gap = paddle.reshape(self.fc1(gap), (b1, c1, 1, 1))  # 1024
        x2_gap = paddle.reshape(self.fc2(gap), (b2, c2, 1, 1))  # 720 + 1170
        # gap=gap.squeeze(-1).transpose((0,2,1)) #bs,1,c
        # x1_gap = self.conv1d(gap)
        # x1_gap = self.sigmoid(x1_gap)
        # x1_gap=x1_gap.transpose((0,2,1)).unsqueeze(-1)

        # x2_gap = self.conv1d(gap)
        # x2_gap = self.sigmoid(x2_gap)
        # x2_gap=x2_gap.transpose((0,2,1)).unsqueeze(-1)

        x1_gap = x1_gap * identity_x1
        x2_gap = x2_gap * identity_x2

        # 空间注意力
        x1_x2_2 = paddle.concat([x1_gap, x2_gap], axis=1)  # 1024 + 720 + 1170

        avg_h = self.pool_h(x1_x2_2)  #
        avg_w = self.pool_w(x1_x2_2)  #

        output_w = self.sigmoid(self.conv_w(avg_w))
        output_h = self.sigmoid(self.conv_h(avg_h))
        # print(output_h.shape)
        l = output_h * output_w  # paddle.transpose(output_w,(0,1,3,2))

        return x1_x2_2 * l


@manager.MODELS.add_component
class server_MMLD(nn.Layer):
    def __init__(self,
                 num_classes,
                 pretrained=None
                 ):
        super(server_MMLD, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        
        self.dem = HRNetW48Contrast(pretrained='./output/pretrained/hrnet_dem.pdparams')
        
        self.fusion = Fusion(720, 1170)
        self.pred = nn.Conv2D(1024 + 64, self.num_classes, 1)
        self.dropout = nn.Dropout2D(0.1)
        self.dem_conv1 = nn.Conv2D(724, 64, kernel_size=1)
        self.dem_conv2 = nn.Conv2D(64, 1, kernel_size=1)
        self.linear_fuse = layers.ConvBNReLU(
            in_channels=1024 + 720 + 1170,
            out_channels=1024,
            kernel_size=1,
            bias_attr=False)
        self.linear_pred = nn.Conv2D(
            1024, self.num_classes, kernel_size=1)

        self.init_weight()

    def forward(self, opt, dem, hill):
        dem_shape = [2, 1, 1024, 1024]
        dem_pred = self.dem(dem)
        
        opt_shape = [2, 1024, 256, 256]
       
        dem2 = F.interpolate(
            dem,
            size=opt_shape[2:],
            mode='bilinear',
            align_corners=False)

        hill2 = F.interpolate(
            hill,
            size=opt_shape[2:],
            mode='bilinear',
            align_corners=False)

        fusion = self.fusion(opt, dem2, hill2)
        fusion = self.dropout(fusion)  # [2, 1808, 256, 256]

        pred = self.linear_fuse(fusion)
        pred = self.linear_pred(pred)
        #
        pred = F.interpolate(
            pred,
            size=dem_shape[2:],
            mode='bilinear',
            align_corners=False)

        return [pred], dem_pred

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


