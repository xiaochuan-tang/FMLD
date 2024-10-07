import paddle.nn as nn
# 导入两个必要的包
from paddleseg.cvlibs import manager
from paddleseg.utils import utils
import paddle
import numpy as np
from paddleseg.cvlibs import manager
from paddleseg.models import layers
import paddle.fluid as fluid
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils


@manager.MODELS.add_component
class HRNetW48Contrast(nn.Layer):
    """
    The HRNetW48Contrast implementation based on PaddlePaddle.

    The original article refers to
    Wenguan Wang, Tianfei Zhou, et al. "Exploring Cross-Image Pixel Contrast for Semantic Segmentation"
    (https://arxiv.org/abs/2101.11939).

    Args:
        in_channels (int): The output dimensions of backbone.
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): Backbone network, currently support HRNet_W48.
        drop_prob (float): The probability of dropout.
        proj_dim (int): The projection dimensions.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 in_channels=720,
                 num_classes=2,
                 backbone=manager.BACKBONES['HRNet_W48'](),
                 drop_prob=0.1,
                 proj_dim=720,
                 align_corners=False,
                 pretrained=None):
        super().__init__()
        self.in_channels = in_channels
        self.backbone = backbone
        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.align_corners = align_corners

        self.cls_head = nn.Sequential(
            layers.ConvBNReLU(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout2D(drop_prob),
            nn.Conv2D(
                in_channels,
                num_classes,
                kernel_size=1,
                stride=1,
                bias_attr=False), )
        self.proj_head = ProjectionHead(
            dim_in=in_channels, proj_dim=self.proj_dim)

        self.dem_pred = nn.Conv2D(
            724, 1, kernel_size=1)

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        feats = self.backbone(x)[0]
        out = self.cls_head(feats)
        # print(out.shape)
        logit_list = []
        if self.training:
            emb = self.proj_head(feats)
            # print(emb.shape)
            logit_list.append(
                F.interpolate(
                    out,
                    paddle.shape(x)[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))
            logit_list.append({'seg': out, 'embed': emb})

            feat1 = F.interpolate(logit_list[0], paddle.shape(x)[2:], mode="bilinear", align_corners=False)
            feat2 = F.interpolate(logit_list[1]['seg'], paddle.shape(x)[2:], mode="bilinear", align_corners=False)
            feat3 = F.interpolate(logit_list[1]['embed'], paddle.shape(x)[2:], mode="bilinear", align_corners=False)
            dem_c = paddle.concat([feat1, feat2, feat3], axis=1)  # [2,724,1024,1024]
            # dem_c = self.dem_pred(dem_c) #[2,1,1024,1024]


        else:
            logit_list.append(
                F.interpolate(
                    out,
                    paddle.shape(x)[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))
            # dem_c=out

            # data = np.zeros((2, 724, 1024, 1024), dtype=np.float32)
            # dem_c = paddle.to_tensor(data)
            dem_c = 1

        # print(logit_list[0].shape)
        # print("logit_list type is {}, and logit_list shape is {}.".format(type(logit_list), paddle.to_tensor(logit_list).shape))
        # print(logit_list[0].shape)  # [2, 2, 1024, 1024]
        # print(logit_list[1]['seg'].shape)   # [2, 2, 256, 256]
        # print(logit_list[1]['embed'].shape) # [2, 720, 256, 256]

        return logit_list, dem_c
        # return feats, dem_c


class ProjectionHead(nn.Layer):
    """
    The projection head used by contrast learning.
    Args:
        dim_in (int): The dimensions of input features.
        proj_dim (int, optional): The output dimensions of projection head. Default: 256.
        proj (str, optional): The type of projection head, only support 'linear' and 'convmlp'. Default: 'convmlp'.
    """

    def __init__(self, dim_in, proj_dim=256, proj='convmlp'):
        super(ProjectionHead, self).__init__()
        if proj == 'linear':
            self.proj = nn.Conv2D(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                layers.ConvBNReLU(
                    dim_in, dim_in, kernel_size=1),
                nn.Conv2D(
                    dim_in, proj_dim, kernel_size=1), )
        else:
            raise ValueError(
                "The type of project head only support 'linear' and 'convmlp', but got {}."
                    .format(proj))

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, axis=1)

