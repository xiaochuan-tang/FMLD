# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from .backbones import *
from .losses import *

from .ann import *
from .bisenet import *
from .danet import *
from .deeplab import *
from .fast_scnn import *
from .fcn import *
from .gcnet import *
from .ocrnet import *
from .pspnet import *
from .gscnn import GSCNN
from .unet import UNet
from .hardnet import HarDNet
from .u2net import U2Net, U2Netp
from .attention_unet import AttentionUNet
from .unet_plusplus import UNetPlusPlus
from .unet_3plus import UNet3Plus
from .decoupled_segnet import DecoupledSegNet
from .emanet import *
from .isanet import *
from .dnlnet import *
from .setr import *
from .sfnet import *
from .pphumanseg_lite import *
from .mla_transformer import MLATransformer
from .portraitnet import PortraitNet
from .stdcseg import STDCSeg
from .segformer import SegFormer
from .pointrend import PointRend
from .ginet import GINet
from .segnet import SegNet

from .dem_feature import Generator
from .sar_feature import Generator_sar
from .dem_label import Slope_net, Aspect, Curvatures

from .mmld_net import MMLD_model
from .segformer_classfusion import SegFormer_class
from .mmld_class_fusion import MMLD_model_class
from .client1_MMLD import SegFormer_B3_client1
from .client2_MMLD import client2_MMLD
# from .client3_MMLD import HighResolutionTransformer
from .server_MMLD import server_MMLD

from .client1_SegFormer import SegFormer_B3_official
from .client2_HRNet import HRNetW48Contrast
from .client3_ResNet import Generator_sar_client3
