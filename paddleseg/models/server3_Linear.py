# # The SegFormer code was heavily based on https://github.com/NVlabs/SegFormer
# # Users should be careful about adopting these functions in any commercial matters.
# # https://github.com/NVlabs/SegFormer#license
#
# import paddle
# import paddle.nn as nn
# import paddle.nn.functional as F
# import numpy as np
#
# from paddleseg.cvlibs import manager
# from paddleseg.models import layers
# from paddleseg.utils import utils
# from paddleseg.models.segformer_classfusion import SegFormer_class_B3, SegFormer_class_B32
# from paddleseg.models.dem_feature import Generator
# from paddleseg.models.sar_feature import Generator_sar
#
#
# # from .backbones import MixVisionTransformer_B4
#
#
# class h_sigmoid(nn.Layer):
#     def __init__(self):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6()
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
#
#
# class SE(nn.Layer):
#     """
#     这个类定义了多模态信息融合模块，用于融合两个输入特征的信息。它包括特征压缩和激活操作，以产生融合后的特征。
#     """
#
#     def __init__(self, in_chnls, ratio):
#         super(SE, self).__init__()
#         self.squeeze = nn.AdaptiveAvgPool2D((1, 1))
#         self.compress = nn.Conv2D(in_chnls, in_chnls // ratio, 1, 1, 0)
#         self.excitation = nn.Conv2D(in_chnls // ratio, in_chnls // 2, 1, 1, 0)
#         # self.conv = nn.Conv2D(in_chnls,in_chnls//2,1,1,0)
#
#     def forward(self, x1, x2):
#         x = paddle.concat([x1, x2], axis=1)
#         # x = self.conv(x1)
#         out = self.squeeze(x)
#         out = self.compress(out)
#         out = F.relu(out)
#         out = self.excitation(out)
#         return x1 * F.sigmoid(out)
#
#
# class Fusion(nn.Layer):
#     def __init__(self, inp1, inp2, reduction=32):
#         super(Fusion, self).__init__()
#         self.pool_h = nn.AdaptiveAvgPool2D((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2D((1, None))
#
#         self.se = SE(inp1 + inp2, 16)  #
#         self.se1 = SE(64 + 64, 16)  #
#
#         mip = max(8, (inp1 + inp2) // reduction)
#
#         self.conv1 = nn.Conv2D(inp1 + inp2, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2D(mip)
#         self.act = h_sigmoid()
#
#         self.conv_h = nn.Conv2D(768 + 64, 1, kernel_size=(7, 1), stride=1, padding=(3, 0))
#
#         self.conv_w = nn.Conv2D(768 + 64, 1, kernel_size=(1, 7), stride=1, padding=(0, 3))
#
#         # self.conv_c = nn.Conv2D(2, 1, kernel_size=7, padding=7 // 2)
#
#         self.gap = nn.AdaptiveAvgPool2D(1)
#         self.conv1d = nn.Conv1D(1, 1, kernel_size=3, padding=1)
#         self.sigmoid = nn.Sigmoid()
#         self.fc1 = nn.Sequential(
#             nn.Conv2D(768 + 64, 768, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2D(768, 768, 1),
#             nn.Sigmoid()
#         )
#         self.fc2 = nn.Sequential(
#             nn.Conv2D(768 + 64, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2D(64, 64, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x12, x21, x22):  # hill,opt,dem,sar
#         # x11.shape: [2, 768, 256, 256] hill
#         # x12.shape: [2, 768, 256, 256] opt
#         # x21.shape: [2, 64, 256, 256]   dem
#         # x22.shape: [2, 64, 256, 256]   sar
#
#         x1 = x12  # hillshade + opt
#         x2 = self.se1(x21, x22)  # dem + sar
#         # x1.shape: [2, 768, 256, 256]
#
#         identity_x1 = x1
#         identity_x2 = x2  # dem 64
#         b1, c1, _, _ = x1.shape
#         b2, c2, _, _ = x2.shape
#         # 通道注意力
#         x1_x2 = paddle.concat([x1, x2], axis=1)
#         # x1_x2.shape: [2, 768 + 64, 256, 256]  [2, 832, 256, 256]
#         gap = self.gap(x1_x2)
#         # gap.shape: [2, 832, 1, 1]
#         x1_gap = paddle.reshape(self.fc1(gap), (b1, c1, 1, 1))
#         x2_gap = paddle.reshape(self.fc2(gap), (b2, c2, 1, 1))
#         # x1_gap.shape: [2, 768, 1, 1]
#         # x2_gap.shape: [2, 64, 1, 1]
#
#         # gap=gap.squeeze(-1).transpose((0,2,1)) #bs,1,c
#         # x1_gap = self.conv1d(gap)
#         # x1_gap = self.sigmoid(x1_gap)
#         # x1_gap=x1_gap.transpose((0,2,1)).unsqueeze(-1)
#
#         # x2_gap = self.conv1d(gap)
#         # x2_gap = self.sigmoid(x2_gap)
#         # x2_gap=x2_gap.transpose((0,2,1)).unsqueeze(-1)
#
#         x1_gap = x1_gap * identity_x1
#         x2_gap = x2_gap * identity_x2
#         # x1_gap.shape: [2, 768, 256, 256]
#         # x2_gap.shape: [2, 64, 256, 256]
#
#         # 空间注意力
#         x1_x2_2 = paddle.concat([x1_gap, x2_gap], axis=1)  #
#         # x1_x2_2.shape: [2, 768 + 64, 256, 256]
#
#         avg_h = self.pool_h(x1_x2_2)  # avg_h.shape: [2, 832, 256, 1]
#         avg_w = self.pool_w(x1_x2_2)  # avg_w.shape: [2, 832, 1, 256]
#
#         output_w = self.sigmoid(self.conv_w(avg_w))  # output_w.shape: [2, 1, 1, 256]
#         output_h = self.sigmoid(self.conv_h(avg_h))  # output_h.shape: [2, 1, 256, 1]
#
#         l = output_h * output_w  # paddle.transpose(output_w,(0,1,3,2))
#         # print("l.shape:", l.shape)    l.shape: [2, 1, 256, 256]
#         # print("x1_x2_2 * l.shape:", (x1_x2_2 * l).shape)  x1_x2_2 * l.shape: [2, 832, 256, 256]
#         return x1_x2_2 * l
#
#
# @manager.MODELS.add_component
# class server3_Linear(nn.Layer):
#     """
#     The SegFormer implementation based on PaddlePaddle.
#
#     The original article refers to
#     Xie, Enze, et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." arXiv preprint arXiv:2105.15203 (2021).
#
#     Args:
#         num_classes (int): The unique number of target classes.
#         backbone (Paddle.nn.Layer): A backbone network.
#         embedding_dim (int): The MLP decoder channel dimension.
#         align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature.
#             is even, e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
#         pretrained (str, optional): The path or url of pretrained model. Default: None.
#     """
#
#     def __init__(self,
#                  num_classes=2,
#                  pretrained=None
#                  ):
#         super(server3_Linear, self).__init__()
#         self.pretrained = pretrained
#         self.num_classes = num_classes
#         # self.opt_seg = SegFormer_class_B3(pretrained='./output/pretrained/mix_vision_transformer_b3_model.pdparams')
#         # self.hill_seg = SegFormer_class_B32(pretrained='./output/pretrained/mix_vision_transformer_b3_model.pdparams')
#         # self.dem = Generator(6, pretrained='./output/pretrained/DEM_old_model.pdparams')
#         # self.sar = Generator_sar(6, pretrained='./output/pretrained/DEM_old_model.pdparams')
#
#         self.fusion = Fusion(768, 768)
#         self.pred = nn.Conv2D(768 + 64, self.num_classes, 1)
#         self.dropout = nn.Dropout2D(0.1)
#         self.linear_fuse = layers.ConvBNReLU(
#             in_channels=64,
#             out_channels=128,
#             kernel_size=1,
#             bias_attr=False)
#         self.linear_pred = nn.Conv2D(
#             128, self.num_classes, kernel_size=1)
#
#         self.init_weight()
#
#     def forward(self, sar):  # 山体阴影图 光学影像 DEM Sar
#         # opt.shape: [2, 768, 256, 256]
#         # dem.shape: [2, 64, 256, 256]
#         # sar.shape: [2, 64, 256, 256]
#
#         # fusion = self.fusion(opt, dem, sar)
#         # # print("fusion.shape:", fusion.shape)  fusion.shape: [2, 768 + 64, 256, 256]
#         # fusion = self.dropout(fusion)
#         # # Tensor(shape=[2, 832, 256, 256], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#
#         pred = self.linear_fuse(sar)
#         pred = self.linear_pred(pred)  # pred.shape: [2, 2, 256, 256]
#         # print("pred.shape:", pred.shape)
#         dem_shape = [2, 2, 1024, 1024]
#         pred = F.interpolate(
#             pred,
#             size=dem_shape[2:],
#             mode='bilinear',
#             align_corners=False)
#         # print("pred_.shape:", pred.shape) [2, 2, 1024, 1024]
#         return [pred]
#
#     def init_weight(self):
#         if self.pretrained is not None:
#             utils.load_entire_model(self, self.pretrained)
#
# # model = MMLD_model(2)
# # x1 = paddle.rand([1, 4, 512, 512])
# # x2 = paddle.rand([1, 3, 512, 512])
# # x3 = paddle.rand([1, 1, 512, 512])
# # out,_ = model(x1,x2,x3)
# # print(out[0].shape)
# # print(paddle.summary(model, [(2, 4, 32, 32),(2, 3, 32, 32),(2, 1, 32, 32)]))










# -*- coding: utf-8 -*-

import paddle.nn as nn
# 导入两个必要的包

from paddleseg.utils import utils
import paddle.nn.functional as F
from paddleseg.cvlibs import manager
from paddle.vision.ops import DeformConv2D as DCN


def swish(x):
    return F.relu(x)


class residualBlock(nn.Layer):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2D(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2D(n)
        self.conv2 = nn.Conv2D(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2D(n)

        self.relu = nn.ReLU()

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = y + x
        return self.relu(y)


@manager.MODELS.add_component
class server3_Linear(nn.Layer):
    def __init__(self, n_residual_blocks=6, pretrained=None, num_classes=2):
        super(server3_Linear, self).__init__()
        self.pretrained = pretrained
        self.n_residual_blocks = n_residual_blocks
        self.conv1 = nn.Conv2D(1, 64, 9, stride=1, padding=4)

        self.resblock = self.make_layer(residualBlock, self.n_residual_blocks)
        # for i in range(self.n_residual_blocks):

        # self.add_module('residual_block' + str(i + 1), residualBlock())

        self.conv2 = nn.Conv2D(64, 64, 3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2D(64)
        self.bn2 = nn.BatchNorm2D(64)
        self.offsets = nn.Conv2D(64, 18, kernel_size=3, stride=1, padding=1)
        self.dconv2_2 = DCN(64, 64, 3, 1, 1)
        self.dconv2_3 = DCN(64, 64, 3, 1, 1)
        self.dconv2_4 = DCN(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2D(64, 2, 1, stride=1, padding=0)
        self.drop = nn.Dropout2D(0.4)
        self.init_weight()

    def make_layer(self, block, num_layers):
        res_block = []
        for i in range(num_layers):
            res_block.append(block())
        return nn.Sequential(*res_block)

    def forward(self, x):
        #########################original version########################
        output = self.conv4(x)
        # print(x.shape)
        # return x
        return [output]
        # return x,self.conv4(x)

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
