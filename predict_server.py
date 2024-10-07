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

import argparse
import os

import paddle
import socket
import time
from paddleseg.cvlibs import manager, Config
from paddleseg.utils import get_sys_env, logger, config_check, get_image_list
from paddleseg.core.predict import predict_server


def parse_args():
    parser = argparse.ArgumentParser(description='Model prediction')

    # params of prediction
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for prediction',
        type=str,
        default=None)
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help=
        'The path of image, it can be a file or a directory including images',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predicted results',
        type=str,
        default='./output/result')

    # augment for prediction
    parser.add_argument(
        '--aug_pred',
        dest='aug_pred',
        help='Whether to use mulit-scales and flip augment for prediction',
        action='store_true')
    parser.add_argument(
        '--scales',
        dest='scales',
        nargs='+',
        help='Scales for augment',
        type=float,
        default=1.0)
    parser.add_argument(
        '--flip_horizontal',
        dest='flip_horizontal',
        help='Whether to use flip horizontally augment',
        action='store_true')
    parser.add_argument(
        '--flip_vertical',
        dest='flip_vertical',
        help='Whether to use flip vertically augment',
        action='store_true')

    # sliding window prediction
    parser.add_argument(
        '--is_slide',
        dest='is_slide',
        help='Whether to prediction by sliding window',
        action='store_true')
    parser.add_argument(
        '--crop_size',
        dest='crop_size',
        nargs=2,
        help=
        'The crop size of sliding window, the first is width and the second is height.',
        type=int,
        default=None)
    parser.add_argument(
        '--stride',
        dest='stride',
        nargs=2,
        help=
        'The stride of sliding window, the first is width and the second is height.',
        type=int,
        default=None)

    # custom color map
    parser.add_argument(
        '--custom_color',
        dest='custom_color',
        nargs='+',
        help=
        'Save images with a custom color map. Default: None, use paddleseg\'s default color map.',
        type=int,
        default=None)
    return parser.parse_args()


def get_test_config(cfg, args):
    test_config = cfg.test_config
    if args.aug_pred:
        test_config['aug_pred'] = args.aug_pred
        test_config['scales'] = args.scales

    if args.flip_horizontal:
        test_config['flip_horizontal'] = args.flip_horizontal

    if args.flip_vertical:
        test_config['flip_vertical'] = args.flip_vertical

    if args.is_slide:
        test_config['is_slide'] = args.is_slide
        test_config['crop_size'] = args.crop_size
        test_config['stride'] = args.stride

    if args.custom_color:
        test_config['custom_color'] = args.custom_color

    return test_config


def main(args, server1, server2, server3):
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(args.cfg)
    val_dataset = cfg.val_dataset
    if not val_dataset:
        raise RuntimeError(
            'The verification dataset is not specified in the configuration file.'
        )

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model
    transforms = val_dataset.transforms
    image_list, image_dir = get_image_list(args.image_path)
    logger.info('Number of predict images = {}'.format(len(image_list)))

    test_config = get_test_config(cfg, args)
    config_check(cfg, val_dataset=val_dataset)

    predict_server(
        model,
        server1,
        server2,
        server3,
        model_path=args.model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=args.save_dir,
        **test_config)


if __name__ == '__main__':
    args = parse_args()

    ########################### init socket###################
    # 基于 Socket 的服务器端程序，用于接收和发送文件数据
    # 在服务器端创建一个套接字，将服务器端套接字绑定到指定的地址和端口
    # socket.AF_INET:表示使用IPv4 地址族;socket.SOCK_STREAM:TCP 协议
    sock1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 通过调用 bind() 方法，服务器套接字与指定的地址和端口进行绑定，以便等待客户端的连接。
    sock1.bind(('127.0.0.1', 9009))

    sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock2.bind(('127.0.0.1', 9010))

    sock3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock3.bind(('127.0.0.1', 9011))

    # sock4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # sock4.bind(('172.26.101.204', 10002))

    # 调用 listen() 方法使这四个 Socket 对象进入监听状态，等待客户端的连接
    print("waiting connection")
    sock1.listen(1)
    sock2.listen(1)
    sock3.listen(1)
    # sock4.listen(1)

    # accept() 方法接受客户端的连接请求，分别返回与客户端连接的 server1、server2、server3、server4 对象和客户端的地址信息
    server1, address1 = sock1.accept()
    server2, address2 = sock2.accept()
    server3, address3 = sock3.accept()
    # server4, address4 = sock4.accept()

    main(args, server1, server2, server3)
