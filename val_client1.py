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

from paddleseg.cvlibs import manager, Config
from paddleseg.core.val_client1 import evaluate
from paddleseg.utils import get_sys_env, logger, config_check, utils
import socket
import time

def recv_cmd(sock_fd):
    cmd = sock_fd.recv(128).decode()
    return cmd


def send_cmd(sock_fd, cmd):
    cmd = str(cmd)
    sock_fd.send(cmd.encode())


def send_file(sock_fd, filename):
    f = open(filename, "rb")
    cont = 0
    while True:
        line = f.read(4096)
        if not line:
            sock_fd.send(bytes("finish".encode()))
            break
        cnt = sock_fd.sendall(line)
    f.close()
    return True


def recv_file(sock_fd, filename):
    f_rev_grad = open(filename, "ab")
    f_rev_grad.seek(0)
    f_rev_grad.truncate()
    while (True):
        buf_rev_grad = sock_fd.recv(4096)
        if buf_rev_grad.find(b"finish") != -1:
            last_buf = buf_rev_grad.replace(b"finish", b"")
            f_rev_grad.write(last_buf)
            f_rev_grad.flush()
            break
        f_rev_grad.write(buf_rev_grad)
        f_rev_grad.flush()
    f_rev_grad.close()
    return True


def get_test_config(cfg, args):
    test_config = cfg.test_config
    if args.aug_eval:
        test_config['aug_eval'] = args.aug_eval
        test_config['scales'] = args.scales

    if args.flip_horizontal:
        test_config['flip_horizontal'] = args.flip_horizontal

    if args.flip_vertical:
        test_config['flip_vertical'] = args.flip_vertical

    if args.is_slide:
        test_config['is_slide'] = args.is_slide
        test_config['crop_size'] = args.crop_size
        test_config['stride'] = args.stride

    return test_config


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')

    # params of evaluate
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for evaluation',
        type=str,
        default=None)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=0)

    # augment for evaluation
    parser.add_argument(
        '--aug_eval',
        dest='aug_eval',
        help='Whether to use mulit-scales and flip augment for evaluation',
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

    # sliding window evaluation
    parser.add_argument(
        '--is_slide',
        dest='is_slide',
        help='Whether to evaluate by sliding window',
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
    parser.add_argument(
        '--data_format',
        dest='data_format',
        help=
        'Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".',
        type=str,
        default='NCHW')

    return parser.parse_args()


def main(args):
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(args.cfg)
    # Only support for the DeepLabv3+ model
    if args.data_format == 'NHWC':
        if cfg.dic['model']['type'] != 'DeepLabV3P':
            raise ValueError(
                'The "NHWC" data format only support the DeepLabV3P model!')
        cfg.dic['model']['data_format'] = args.data_format
        cfg.dic['model']['backbone']['data_format'] = args.data_format
        loss_len = len(cfg.dic['loss']['types'])
        for i in range(loss_len):
            cfg.dic['loss']['types'][i]['data_format'] = args.data_format

    val_dataset = cfg.val_dataset
    if val_dataset is None:
        raise RuntimeError(
            'The verification dataset is not specified in the configuration file.'
        )
    elif len(val_dataset) == 0:
        raise ValueError(
            'The length of val_dataset is 0. Please check if your dataset is valid'
        )

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model
    if args.model_path:
        utils.load_entire_model(model, args.model_path)
        logger.info('Loaded trained params of model successfully')

    test_config = get_test_config(cfg, args)
    config_check(cfg, val_dataset=val_dataset)

    # 在客户端创建一个套接字，将客户端套接字连接到指定的服务器地址和端口
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('127.0.0.1', 9011))
    print("already connect")
    time.sleep(2)

    evaluate(model, val_dataset, client, num_workers=args.num_workers, **test_config)


if __name__ == '__main__':
    args = parse_args()
    main(args)
