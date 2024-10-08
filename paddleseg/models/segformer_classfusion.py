# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import numpy as np

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils


class MLP(nn.Layer):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.proj(x)
        return x


@manager.MODELS.add_component
class SegFormer_class(nn.Layer):
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
                 backbone,
                 embedding_dim,
                 align_corners=False,
                 pretrained=None):
        super(SegFormer_class, self).__init__()

        self.pretrained = pretrained
        self.align_corners = align_corners
        self.backbone = backbone
        # self.num_classes = num_classes
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.backbone.feat_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.dropout = nn.Dropout2D(0.1)
        self.linear_fuse = layers.ConvBNReLU(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            bias_attr=False)


        # self.linear_pred = nn.Conv2D(
        #     embedding_dim, self.num_classes, kernel_size=1)

        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        feats = self.backbone(x)
        c1, c2, c3, c4 = feats

        ############## MLP decoder on C1-C4 ###########
        c1_shape = paddle.shape(c1)
        c2_shape = paddle.shape(c2)
        c3_shape = paddle.shape(c3)
        c4_shape = paddle.shape(c4)

        # _c4 = F.interpolate(
        #     c4,
        #     size=c1_shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # _c3 = F.interpolate(
        #     c3,
        #     size=c1_shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # _c2 = F.interpolate(
        #     c2,
        #     size=c1_shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # _c1 = F.interpolate(
        #     c1,
        #     size=c1_shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # print(c1_shape,c2_shape,c3_shape,c4_shape)
        _c4 = self.linear_c4(c4).transpose([0, 2, 1]).reshape(
            [0, 0, c4_shape[2], c4_shape[3]])
        _c4 = F.interpolate(
            _c4,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).transpose([0, 2, 1]).reshape(
            [0, 0, c3_shape[2], c3_shape[3]])
        _c3 = F.interpolate(
            _c3,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).transpose([0, 2, 1]).reshape(
            [0, 0, c2_shape[2], c2_shape[3]])
        _c2 = F.interpolate(
            _c2,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).transpose([0, 2, 1]).reshape(
            [0, 0, c1_shape[2], c1_shape[3]])

        _c = paddle.concat([_c4, _c3, _c2, _c1], axis=1)
        _c = self.linear_fuse(_c)

        # logit = self.dropout(_c)
        # logit = self.linear_pred(logit)
        # return [
        #     F.interpolate(
        #         logit,
        #         size=paddle.shape(x)[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        # ]
        return [_c]


@manager.MODELS.add_component
def SegFormer_class_B0(**kwargs):
    return SegFormer_class(
        backbone=manager.BACKBONES['MixVisionTransformer_B0'](),
        embedding_dim=768,
        **kwargs)


@manager.MODELS.add_component
def SegFormer_class_B1(**kwargs):
    return SegFormer_class(
        backbone=manager.BACKBONES['MixVisionTransformer_B1'](),
        embedding_dim=256,
        **kwargs)


@manager.MODELS.add_component
def SegFormer_class_B2(**kwargs):
    return SegFormer_class(
        backbone=manager.BACKBONES['MixVisionTransformer_B2'](),
        embedding_dim=768,
        **kwargs)


@manager.MODELS.add_component
def SegFormer_class_B3(**kwargs):
    return SegFormer_class(
        backbone=manager.BACKBONES['MixVisionTransformer_B3'](),
        embedding_dim=768,
        **kwargs)

@manager.MODELS.add_component
def SegFormer_class_B32(**kwargs):
    return SegFormer_class(
        backbone=manager.BACKBONES['MixVisionTransformer_B32'](),
        embedding_dim=768,
        **kwargs)

@manager.MODELS.add_component
def SegFormer_class_B4(**kwargs):
    return SegFormer_class(
        backbone=manager.BACKBONES['MixVisionTransformer_B4'](),
        embedding_dim=768,
        **kwargs)

@manager.MODELS.add_component
def SegFormer_class_B5(**kwargs):
    return SegFormer_class(
        backbone=manager.BACKBONES['MixVisionTransformer_B5'](),
        embedding_dim=768,
        **kwargs)
