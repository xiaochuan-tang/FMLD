# The SegFormer code was heavily based on https://github.com/NVlabs/SegFormer
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/NVlabs/SegFormer#license

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils
from paddleseg.models.segformer import SegFormer_B3, SegFormer_B32
from paddleseg.models.dem_feature import Generator


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
        self.excitation = nn.Conv2D(in_chnls // ratio, in_chnls // 2, 1, 1, 0)
        # self.conv = nn.Conv2D(in_chnls,in_chnls//2,1,1,0)

    def forward(self, x1, x2):
        x = paddle.concat([x1, x2], axis=1)
        # x = self.conv(x1)
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return x1 * F.sigmoid(out)


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

        self.conv_h = nn.Conv2D(1024 + 64, 1, kernel_size=(7, 1), stride=1, padding=(3, 0))

        self.conv_w = nn.Conv2D(1024 + 64, 1, kernel_size=(1, 7), stride=1, padding=(0, 3))

        # self.conv_c = nn.Conv2D(2, 1, kernel_size=7, padding=7 // 2)

        self.gap = nn.AdaptiveAvgPool2D(1)
        self.conv1d = nn.Conv1D(1, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Sequential(
            nn.Conv2D(1024 + 64, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Conv2D(1024, 1024, 1),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Conv2D(1024 + 64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2D(64, 64, 1),
            nn.Sigmoid()
        )

    def forward(self, x11, x12, x2):  # hill,opt,dem
        x1 = self.se(x11, x12)  # hillshade+opt
        identity_x1 = x1
        identity_x2 = x2  # dem 64
        b1, c1, _, _ = x1.shape
        b2, c2, _, _ = x2.shape
        # 通道注意力
        x1_x2 = paddle.concat([x1, x2], axis=1)
        gap = self.gap(x1_x2)
        x1_gap = paddle.reshape(self.fc1(gap), (b1, c1, 1, 1))
        x2_gap = paddle.reshape(self.fc2(gap), (b2, c2, 1, 1))
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
        x1_x2_2 = paddle.concat([x1_gap, x2_gap], axis=1)  #

        avg_h = self.pool_h(x1_x2_2)  #
        avg_w = self.pool_w(x1_x2_2)  #

        output_w = self.sigmoid(self.conv_w(avg_w))
        output_h = self.sigmoid(self.conv_h(avg_h))
        # print(output_h.shape)
        l = output_h * output_w  # paddle.transpose(output_w,(0,1,3,2))

        return x1_x2_2 * l


@manager.MODELS.add_component
class MMLD_model(nn.Layer):
    """
    The SegFormer implementation based on PaddlePaddle.

    The original article refers to
    Xie, Enze, et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." arXiv preprint arXiv:2105.15203 (2021).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): A backbone network.
        embedding_dim (int): The MLP decoder channel dimension.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature.
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 pretrained=None
                 ):
        super(MMLD_model, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.opt_seg = SegFormer_B3(pretrained='./output/pretrained/mix_vision_transformer_b3_model.pdparams')
        self.hill_seg = SegFormer_B32(pretrained='./output/pretrained/mix_vision_transformer_b3_model.pdparams')
        self.dem = Generator(6, pretrained='./output/pretrained/DEM_old_model.pdparams')

        self.fusion = Fusion(1024, 1024)
        self.pred = nn.Conv2D(1024 + 64, self.num_classes, 1)
        self.dropout = nn.Dropout2D(0.1)
        self.linear_fuse = layers.ConvBNReLU(
            in_channels=1024 + 64,
            out_channels=1024,
            kernel_size=1,
            bias_attr=False)
        self.linear_pred = nn.Conv2D(
            1024, self.num_classes, kernel_size=1)

        self.init_weight()

    def forward(self, hill, opt, dem, sar):
        # print(dem)
        # hill = paddle.concat([hill,dem],axis=1)
        # opt = paddle.concat([opt, dem], axis=1)

        hill = self.hill_seg(hill)  # hill type is <class 'list'>, and hill shape is [1, 2, 1024, 256, 256].
        opt = self.opt_seg(opt)     # opt type is <class 'list'>, and opt shape is [1, 2, 1024, 256, 256].

        # print("hill type is {}, and hill shape is {}.".format(type(hill), paddle.to_tensor(hill).shape))
        # print("opt type is {}, and opt shape is {}.".format(type(opt), paddle.to_tensor(opt).shape))

        dem_shape = paddle.shape(dem)
        hill_shape = paddle.shape(hill[0])

        # hill_opt = paddle.concat([hill[0],opt[0]],axis=1)
        # fusion = self.linear_fuse(hill_opt)
        # pred = self.linear_pred(fusion)

        dem, dem_pred = self.dem(dem)
        dem_pred.stop_gradient = True

        dem2 = F.interpolate(
            dem,
            size=hill_shape[2:],
            mode='bilinear',
            align_corners=False)

        fusion = self.fusion(hill[0], opt[0], dem2)
        fusion = self.dropout(fusion)
        # print(fusion.shape)
        # pred = self.pred(fusion)
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

# model = MMLD_model(2)
# x1 = paddle.rand([1, 4, 512, 512])
# x2 = paddle.rand([1, 3, 512, 512])
# x3 = paddle.rand([1, 1, 512, 512])
# out,_ = model(x1,x2,x3)
# print(out[0].shape)
# print(paddle.summary(model, [(2, 4, 32, 32),(2, 3, 32, 32),(2, 1, 32, 32)]))
