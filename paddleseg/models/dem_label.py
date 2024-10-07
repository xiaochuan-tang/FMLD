import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import math
from paddleseg.cvlibs import manager

@manager.MODELS.add_component
class Slope_net(nn.Layer):
    def __init__(self):
        super(Slope_net, self).__init__()
        weight1 = np.zeros(shape=(3, 3), dtype=np.float32)
        weight2 = np.zeros(shape=(3, 3), dtype=np.float32)

        weight1[0][0] = -1
        weight1[0][1] = 0
        weight1[0][2] = 1
        weight1[1][0] = -2
        weight1[1][1] = 0
        weight1[1][2] = 2
        weight1[2][0] = -1
        weight1[2][1] = 0
        weight1[2][2] = 1

        weight2[0][0] = -1
        weight2[0][1] = -2
        weight2[0][2] = -1
        weight2[1][0] = 0
        weight2[1][1] = 0
        weight2[1][2] = 0
        weight2[2][0] = 1
        weight2[2][1] = 2
        weight2[2][2] = 1
        weight1 = np.reshape(weight1, (1, 1, 3, 3))
        weight2 = np.reshape(weight2, (1, 1, 3, 3))
        weight1 = weight1 / (8 * 0.5)
        weight2 = weight2 / (8 * 0.5)
        self.weight1 = paddle.to_tensor(weight1).astype('float32')
        self.weight2 = paddle.to_tensor(weight2).astype('float32')
        self.padd = nn.Pad2D(1, mode='replicate')
        # self.bias = paddle.zeros([1],dtype='float32')
        # self.weight1 =paddle.static.create_parameter([1,1,3,3],dtype='float32',default_initializer = weight1)  # 自定义的权值
        # self.weight2 = paddle.static.create_parameter([1,1,3,3],dtype='float32',default_initializer = weight2)
        # self.bias = paddle.static.create_parameter(paddle.zeros(1),dtype='float32')  # 自定义的偏置

    def forward(self, x):
        x = self.padd(x)
        dx = F.conv2d(x, self.weight1, stride=1, padding=0)
        dy = F.conv2d(x, self.weight2, stride=1, padding=0)
        ij_slope = paddle.sqrt(paddle.pow(dx, 2) + paddle.pow(dy, 2))
        ij_slope = paddle.atan(ij_slope) * 180 / math.pi
        return ij_slope


@manager.MODELS.add_component
class Curvatures(nn.Layer):
    def __init__(self):
        super(Curvatures, self).__init__()
        weight1 = np.zeros(shape=(3, 3), dtype=np.float32)
        weight2 = np.zeros(shape=(3, 3), dtype=np.float32)

        weight1[0][0] = -1
        weight1[0][1] = 0
        weight1[0][2] = 1
        weight1[1][0] = -2
        weight1[1][1] = 0
        weight1[1][2] = 2
        weight1[2][0] = -1
        weight1[2][1] = 0
        weight1[2][2] = 1

        weight2[0][0] = -1
        weight2[0][1] = -2
        weight2[0][2] = -1
        weight2[1][0] = 0
        weight2[1][1] = 0
        weight2[1][2] = 0
        weight2[2][0] = 1
        weight2[2][1] = 2
        weight2[2][2] = 1
        weight1 = np.reshape(weight1, (1, 1, 3, 3))
        weight2 = np.reshape(weight2, (1, 1, 3, 3))
        weight1 = weight1 / (8 * 0.5)
        weight2 = weight2 / (8 * 0.5)
        self.weight1 = paddle.to_tensor(weight1).astype('float32')
        self.weight2 = paddle.to_tensor(weight2).astype('float32')
        self.padd = nn.Pad2D(1, mode='replicate')
        # self.bias = paddle.zeros([1],dtype='float32')
        # self.weight1 =paddle.static.create_parameter([1,1,3,3],dtype='float32',default_initializer = weight1)  # 自定义的权值
        # self.weight2 = paddle.static.create_parameter([1,1,3,3],dtype='float32',default_initializer = weight2)
        # self.bias = paddle.static.create_parameter(paddle.zeros(1),dtype='float32')  # 自定义的偏置

    def forward(self, x):
        x = self.padd(x)
        p = F.conv2d(x, self.weight1, stride=1, padding=0)
        q = F.conv2d(x, self.weight2, stride=1, padding=0)
        p2 = self.padd(p)
        q2 = self.padd(q)
        r = F.conv2d(p2, self.weight1, stride=1, padding=0)
        s = F.conv2d(p2, self.weight2, stride=1, padding=0)
        t = F.conv2d(q2, self.weight2, stride=1, padding=0)

        Kv = -(p * p * r + 2 * p * q * s + q * q * t) / ((p * p + q * q) * pow((1 + p * p + q * q), 3 / 2)+1e-6)  # 剖面曲率
        Km = - ((1 + q * q) * r - 2 * s * p * q + (1 + p * p) * t) / (2 * pow((1 + p * p + q * q), 3 / 2))  # 平均曲率
        # Kv[paddle.isnan(Kv)] = 0
        # Kv = paddle.where(paddle.isnan(Kv),0,Kv)
        return Kv


@manager.MODELS.add_component
class Aspect(nn.Layer):

    def __init__(self):
        super(Aspect, self).__init__()
        weight1 = np.zeros(shape=(3, 3), dtype=np.float32)
        weight2 = np.zeros(shape=(3, 3), dtype=np.float32)

        weight1[0][0] = -1
        weight1[0][1] = 0
        weight1[0][2] = 1
        weight1[1][0] = -2
        weight1[1][1] = 0
        weight1[1][2] = 2
        weight1[2][0] = -1
        weight1[2][1] = 0
        weight1[2][2] = 1

        weight2[0][0] = -1
        weight2[0][1] = -2
        weight2[0][2] = -1
        weight2[1][0] = 0
        weight2[1][1] = 0
        weight2[1][2] = 0
        weight2[2][0] = 1
        weight2[2][1] = 2
        weight2[2][2] = 1

        weight1 = np.reshape(weight1, (1, 1, 3, 3))
        weight2 = np.reshape(weight2, (1, 1, 3, 3))
        weight1 = weight1 / (8 * 0.5)
        weight2 = weight2 / (8 * 0.5)
        self.weight1 = paddle.to_tensor(weight1).astype('float32')
        self.weight2 = paddle.to_tensor(weight2).astype('float32')
        self.padd = nn.Pad2D(1, mode='replicate')
        # self.weight1 = nn.Parameter(torch.tensor(weight1))  # 自定义的权值
        # self.weight2 = nn.Parameter(torch.tensor(weight2))
        # self.bias = nn.Parameter(torch.zeros(1))  # 自定义的偏置

    def forward(self, x):
        x = self.padd(x)
        dx = F.conv2d(x, self.weight1, stride=1, padding=0)
        dy = F.conv2d(x, self.weight2, stride=1, padding=0)
        batchsize, channel, row, col = x.shape
        aspect = 57.29578 * paddle.atan2(dy, -dx)
        # a = np.array(aspect[1][0].cpu())
        aspect = paddle.where(aspect > 90, 360 - aspect + 90, 90 - aspect)
        return aspect


class Slope2(nn.Layer):
    def __init__(self):
        super(Slope2, self).__init__()
        weight1 = np.zeros(shape=(7, 7), dtype=np.float32)
        weight2 = np.zeros(shape=(7, 7), dtype=np.float32)

        weight1[0][0] = -1
        weight1[0][6] = 1

        weight1[3][0] = -2
        weight1[3][6] = 2

        weight1[6][0] = -1
        weight1[6][6] = 1

        weight2[0][0] = -1
        weight2[0][3] = -2
        weight2[0][6] = -1

        weight2[6][0] = 1
        weight2[6][3] = 2
        weight2[6][6] = 1

        weight1 = np.reshape(weight1, (1, 1, 7, 7))
        weight2 = np.reshape(weight2, (1, 1, 7, 7))
        weight1 = weight1 / (8 * 0.5)
        weight2 = weight2 / (8 * 0.5)
        self.weight1 = paddle.to_tensor(weight1).astype('float32')
        self.weight2 = paddle.to_tensor(weight2).astype('float32')
        self.padd = nn.Pad2D(3, mode='replicate')
        # self.bias = paddle.zeros([1],dtype='float32')
        # self.weight1 =paddle.static.create_parameter([1,1,3,3],dtype='float32',default_initializer = weight1)  # 自定义的权值
        # self.weight2 = paddle.static.create_parameter([1,1,3,3],dtype='float32',default_initializer = weight2)
        # self.bias = paddle.static.create_parameter(paddle.zeros(1),dtype='float32')  # 自定义的偏置

    def forward(self, x):
        x = self.padd(x)
        dx = F.conv2d(x, self.weight1, stride=1, padding=0)
        dy = F.conv2d(x, self.weight2, stride=1, padding=0)
        ij_slope = paddle.sqrt(paddle.pow(dx, 2) + paddle.pow(dy, 2))
        ij_slope = paddle.atan(ij_slope) * 180 / math.pi
        return ij_slope


class Aspect2(nn.Layer):
    def __init__(self):
        super(Aspect2, self).__init__()
        weight1 = np.zeros(shape=(7, 7), dtype=np.float32)
        weight2 = np.zeros(shape=(7, 7), dtype=np.float32)

        weight1[0][0] = -1
        weight1[0][6] = 1

        weight1[3][0] = -2
        weight1[3][6] = 2

        weight1[6][0] = -1
        weight1[6][6] = 1

        weight2[0][0] = -1
        weight2[0][3] = -2
        weight2[0][6] = -1

        weight2[6][0] = 1
        weight2[6][3] = 2
        weight2[6][6] = 1

        weight1 = np.reshape(weight1, (1, 1, 7, 7))
        weight2 = np.reshape(weight2, (1, 1, 7, 7))
        weight1 = weight1 / (8 * 0.5)
        weight2 = weight2 / (8 * 0.5)
        self.weight1 = paddle.to_tensor(weight1).astype('float32')
        self.weight2 = paddle.to_tensor(weight2).astype('float32')
        self.padd = nn.Pad2D(3, mode='replicate')
        # self.weight1 = nn.Parameter(torch.tensor(weight1))  # 自定义的权值
        # self.weight2 = nn.Parameter(torch.tensor(weight2))
        # self.bias = nn.Parameter(torch.zeros(1))  # 自定义的偏置

    def forward(self, x):
        x = self.padd(x)
        dx = F.conv2d(x, self.weight1, stride=1, padding=0)
        dy = F.conv2d(x, self.weight2, stride=1, padding=0)
        batchsize, channel, row, col = x.shape
        aspect = 57.29578 * paddle.atan2(dy, -dx)
        # a = np.array(aspect[1][0].cpu())
        aspect = paddle.where(aspect > 90, 360 - aspect + 90, 90 - aspect)
        return aspect


@manager.MODELS.add_component
class Line(nn.Layer):
    def __init__(self):
        super(Line, self).__init__()
        self.slope = Slope2()
        self.aspect = Aspect2()
        self.meanpool = nn.AvgPool2D(kernel_size=5, stride=5)
        # self.maxpool = nn.MaxPool2d(kernel_size=12, stride=12)

    def forward(self, x):
        b, c, w, h = x.shape

        max_ = paddle.max(x)
        # max = F.interpolate(max, size=(w, h), mode='nearest')
        FDEM = max_ - x
        # print(FDEM)
        FDEM1 = self.aspect(FDEM)
        SOA2 = self.slope(FDEM1)

        DEM1 = self.aspect(x)
        SOA1 = self.slope(DEM1)
        SOA = ((SOA1 + SOA2) - paddle.abs(SOA1 - SOA2)) / 2

        mean = self.meanpool(x)
        mean = F.interpolate(mean, paddle.shape(x)[2:], mode='bilinear', align_corners=False)
        ZFDX = x - mean
        ZFDX = np.array(ZFDX)
        SOA = np.array(SOA)
        shanji_line = np.where((ZFDX > 0.7) & (SOA > 85), 2, 0)
        shangu_line = np.where((ZFDX < -0.8) & (SOA > 85), 1, 0)
        line = abs(shanji_line - shangu_line)  # 2表示山脊 1表示山谷
        line = paddle.to_tensor(line)
        return line
