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
from paddleseg.models.segformer_classfusion import SegFormer_class_B3, SegFormer_class_B32
from paddleseg.models.dem_feature import Generator
from paddleseg.models.sar_feature import Generator_sar
from paddleseg.models.backbones import HRFormer_small
from paddleseg.models.backbones.resnet_vd import ResNet101_vd


# from .backbones import MixVisionTransformer_B4


class h_sigmoid(nn.Layer):
    def __init__(self):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6


class SE(nn.Layer):
    """
    这个类定义了多模态信息融合模块，用于融合两个输入特征的信息。它包括特征压缩和激活操作，以产生融合后的特征。
    """

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
        self.se1 = SE(64 + 64, 16)  #

        mip = max(8, (inp1 + inp2) // reduction)

        self.conv1 = nn.Conv2D(inp1 + inp2, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2D(mip)
        self.act = h_sigmoid()

        self.conv_h = nn.Conv2D(768 + 64, 1, kernel_size=(7, 1), stride=1, padding=(3, 0))

        self.conv_w = nn.Conv2D(768 + 64, 1, kernel_size=(1, 7), stride=1, padding=(0, 3))

        # self.conv_c = nn.Conv2D(2, 1, kernel_size=7, padding=7 // 2)

        self.gap = nn.AdaptiveAvgPool2D(1)
        self.conv1d = nn.Conv1D(1, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Sequential(
            nn.Conv2D(768 + 64, 768, 3, padding=1),
            nn.ReLU(),
            nn.Conv2D(768, 768, 1),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Conv2D(768 + 64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2D(64, 64, 1),
            nn.Sigmoid()
        )

    def forward(self, x12, x21, x22):  # hill,opt,dem,sar
        # x11.shape: [2, 768, 256, 256] hill
        # x12.shape: [2, 768, 256, 256] opt
        # x21.shape: [2, 64, 256, 256]   dem
        # x22.shape: [2, 64, 256, 256]   sar

        # x1 = self.se(x11, x12)  # hillshade + opt
        x1 = x12
        x2 = self.se1(x21, x22)  # dem + sar
        # x1.shape: [2, 768, 256, 256]

        identity_x1 = x1
        identity_x2 = x2  # dem 64
        b1, c1, _, _ = x1.shape
        b2, c2, _, _ = x2.shape
        # 通道注意力
        x1_x2 = paddle.concat([x1, x2], axis=1)
        # x1_x2.shape: [2, 768 + 64, 256, 256]  [2, 832, 256, 256]
        gap = self.gap(x1_x2)
        # gap.shape: [2, 832, 1, 1]
        x1_gap = paddle.reshape(self.fc1(gap), (b1, c1, 1, 1))
        x2_gap = paddle.reshape(self.fc2(gap), (b2, c2, 1, 1))
        # x1_gap.shape: [2, 768, 1, 1]
        # x2_gap.shape: [2, 64, 1, 1]

        # gap=gap.squeeze(-1).transpose((0,2,1)) #bs,1,c
        # x1_gap = self.conv1d(gap)
        # x1_gap = self.sigmoid(x1_gap)
        # x1_gap=x1_gap.transpose((0,2,1)).unsqueeze(-1)

        # x2_gap = self.conv1d(gap)
        # x2_gap = self.sigmoid(x2_gap)
        # x2_gap=x2_gap.transpose((0,2,1)).unsqueeze(-1)

        x1_gap = x1_gap * identity_x1
        x2_gap = x2_gap * identity_x2
        # x1_gap.shape: [2, 768, 256, 256]
        # x2_gap.shape: [2, 64, 256, 256]

        # 空间注意力
        x1_x2_2 = paddle.concat([x1_gap, x2_gap], axis=1)  #
        # x1_x2_2.shape: [2, 768 + 64, 256, 256]

        avg_h = self.pool_h(x1_x2_2)  # avg_h.shape: [2, 832, 256, 1]
        avg_w = self.pool_w(x1_x2_2)  # avg_w.shape: [2, 832, 1, 256]

        output_w = self.sigmoid(self.conv_w(avg_w))  # output_w.shape: [2, 1, 1, 256]
        output_h = self.sigmoid(self.conv_h(avg_h))  # output_h.shape: [2, 1, 256, 1]

        l = output_h * output_w  # paddle.transpose(output_w,(0,1,3,2))
        # print("l.shape:", l.shape)    l.shape: [2, 1, 256, 256]
        # print("x1_x2_2 * l.shape:", (x1_x2_2 * l).shape)  x1_x2_2 * l.shape: [2, 832, 256, 256]
        return x1_x2_2 * l


@manager.MODELS.add_component
class MMLD_model_class(nn.Layer):
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
        super(MMLD_model_class, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.opt_seg = SegFormer_class_B3(pretrained='./output/pretrained/mix_vision_transformer_b3_model.pdparams')
        self.hill_seg = SegFormer_class_B32(pretrained='./output/pretrained/mix_vision_transformer_b3_model.pdparams')
        # self.dem = Generator(6, pretrained='./output/pretrained/DEM_old_model.pdparams')
        self.dem = HRFormer_small(in_channels=1, pretrained="./output/pretrained/hill_mse_hrformer_small.pdparams")
        # self.sar = Generator_sar(6, pretrained='./output/pretrained/DEM_old_model.pdparams')
        self.sar = ResNet101_vd(pretrained='./output/pretrained/DEM_old_model.pdparams')

        self.fusion = Fusion(768, 768)
        self.pred = nn.Conv2D(768 + 64, self.num_classes, 1)
        self.dropout = nn.Dropout2D(0.1)
        self.linear_fuse = layers.ConvBNReLU(
            in_channels=768 + 64,
            out_channels=768,
            kernel_size=1,
            bias_attr=False)
        self.linear_pred = nn.Conv2D(
            768, self.num_classes, kernel_size=1)

        self.conv = paddle.nn.Conv2D(in_channels=224, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.init_weight()

    def forward(self, hill, opt, dem, sar):  # 山体阴影图 光学影像 DEM Sar
        # print(dem)

        # hill.shape: [2, 3, 1024, 1024]
        # opt.shape: [2, 3, 1024, 1024]
        # dem.shape: [2, 1, 1024, 1024]
        # sar.shape: [2, 1, 1024, 1024]

        # hill = paddle.concat([hill,dem],axis=1)
        # opt = paddle.concat([opt, dem], axis=1)

        # hill = self.hill_seg(hill)  # 1024
        # hill type is <class 'list'>, and hill shape is [1, 2, 768, 256, 256].
        # print("hill type is {}, and hill shape is {}.".format(type(hill), paddle.to_tensor(hill).shape))
        opt = self.opt_seg(opt)
        # opt type is <class 'list'>, and opt shape is [1, 2, 768, 256, 256].
        # print("opt type is {}, and opt shape is {}.".format(type(opt), paddle.to_tensor(opt).shape))

        dem_shape = paddle.shape(dem)  # Tensor [2,1,1024,1024]
        opt_shape = paddle.shape(opt[0])  # Tensor [2, 768, 256, 256]

        dem_p = self.dem(dem)  # dem.shape: [2, 64, 1024, 1024]
        # print("dem type is {}, and dem shape is {}.".format(type(dem), paddle.to_tensor(dem).shape))
        # dem_pred.stop_gradient = True
        dem_shape_after = [2, 64, 256, 256]

        feat1 = F.interpolate(
            dem_p[0], dem_shape_after[2:], mode="bilinear", align_corners=False)
        feat2 = F.interpolate(
            dem_p[1], dem_shape_after[2:], mode="bilinear", align_corners=False)
        feat3 = F.interpolate(
            dem_p[2], dem_shape_after[2:], mode="bilinear", align_corners=False)
        dem_c = paddle.concat([feat1, feat2, feat3], axis=1)
        dem_c = self.conv(dem_c)

        # dem2 = F.interpolate(
        #     dem,
        #     size=opt_shape[2:],
        #     mode='bilinear',
        #     align_corners=False)
        # # dem2.shape: [2, 64, 256, 256]

        sar = self.sar(sar)
        sar2 = F.interpolate(
            sar,
            size=opt_shape[2:],
            mode='bilinear',
            align_corners=False)

        fusion = self.fusion(opt[0], dem_c, sar2)
        # print("fusion.shape:", fusion.shape)  fusion.shape: [2, 768 + 64, 256, 256]
        fusion = self.dropout(fusion)
        # print(fusion.shape)
        # print(fusion)
        # pred = self.pred(fusion)
        pred = self.linear_fuse(fusion)
        pred = self.linear_pred(pred)  # pred.shape: [2, 2, 256, 256]
        # print("pred.shape:", pred.shape)
        pred = F.interpolate(
            pred,
            size=dem_shape[2:],
            mode='bilinear',
            align_corners=False)
        # print("pred_.shape:", pred.shape) [2, 2, 1024, 1024]
        # return [pred], dem_pred
        return [pred]

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
