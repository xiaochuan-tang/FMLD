# import paddle
#
# print("hello1")
# # 定义网络和损失函数
# class Net(paddle.nn.Layer):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.linear = paddle.nn.Linear(2, 1)
#
#     def forward(self, x):
#         return self.linear(x)
#
# net = Net()
# criterion = paddle.nn.MSELoss()
#
# # 定义输入数据和目标数据
# x_data = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
# y_data = paddle.to_tensor([[3.0], [7.0]])
#
# # 前向传播
# y_pred = net(x_data)
# loss = criterion(y_pred, y_data)
#
# # 反向传播
# loss.backward()
#
# # print(net.grad)
#
# # 获取梯度
# for param in net.parameters():
#     if param.grad is not None:
#         print(param.grad)
#
# import paddle
#
# x = paddle.to_tensor(5., stop_gradient=False)
# y = paddle.pow(x, 4.0)
# y.backward()
# print("grad of x: {}".format(x.grad))
# # Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=False, [500.])
#
# import paddle
#
# x = paddle.randn([3, 3], dtype='float32')
# print("Before detach, stop_gradient:", x.stop_gradient)  # 输出 False
#
# # 创建一个新的不需要梯度的张量副本
# x_stop_gradient = x.detach()
# x_stop_gradient.stop_gradient = False
#
# print("After detach, stop_gradient:", x_stop_gradient.stop_gradient)  # 输出 True

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class AttentionModule(nn.Layer):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2D(in_channels, 1, 1)

    def forward(self, x):
        # 计算通道注意力权重
        weights = F.sigmoid(self.conv(x))
        # 加权求和
        x = x * weights
        # 求和后的特征图
        x = paddle.sum(x, axis=1, keepdim=True)
        return x


# 假设有三个特征图，每个特征图的形状为(1, 64, 256, 256)
feat1 = paddle.randn([1, 64, 256, 256])
feat2 = paddle.randn([1, 64, 256, 256])
feat3 = paddle.randn([1, 64, 256, 256])

# 将三个特征图拼接在一起
feat_cat = paddle.concat([feat1, feat2, feat3], axis=1)

# 使用注意力机制融合特征图
attention_module = AttentionModule(in_channels=3 * 64)
feat_fused = attention_module(feat_cat)

print(feat_fused.shape)  # 输出特征图的形状
