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
import random

import paddle
import numpy as np
import socket
from paddleseg.cvlibs import manager, Config
from paddleseg.utils import get_sys_env, logger, config_check
from paddleseg.core import train_server


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--iters',
        dest='iters',
        help='iters for training',
        type=int,
        default=None)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu',
        type=int,
        default=None)
    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        help='Learning rate',
        type=float,
        default=None)
    parser.add_argument(
        '--save_interval',
        dest='save_interval',
        help='How many iters to save a model snapshot once during training.',
        type=int,
        default=1000)
    parser.add_argument(
        '--resume_model',
        dest='resume_model',
        help='The path of resume model',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output')
    parser.add_argument(
        '--keep_checkpoint_max',
        dest='keep_checkpoint_max',
        help='Maximum number of checkpoints to save',
        type=int,
        default=5)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=0)
    parser.add_argument(
        '--do_eval',
        dest='do_eval',
        help='Eval while training',
        action='store_true')
    parser.add_argument(
        '--log_iters',
        dest='log_iters',
        help='Display logging information at every log_iters',
        default=10,
        type=int)
    parser.add_argument(
        '--use_vdl',
        dest='use_vdl',
        help='Whether to record the data to VisualDL during training',
        action='store_true')
    parser.add_argument(
        '--seed',
        dest='seed',
        help='Set the random seed during training.',
        default=None,
        type=int)
    parser.add_argument(
        '--fp16', dest='fp16', help='Whther to use amp', action='store_true')
    parser.add_argument(
        '--data_format',
        dest='data_format',
        help=
        'Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".',
        type=str,
        default='NCHW')
    parser.add_argument(
        '--profiler_options',
        type=str,
        default=None,
        help='The option of train profiler. If profiler_options is not None, the train ' \
             'profiler is enabled. Refer to the paddleseg/utils/train_profiler.py for details.'
    )

    return parser.parse_args()


def main(args, server1, server2, server3):
    if args.seed is not None:
        paddle.seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(
        args.cfg,
        learning_rate=args.learning_rate,
        iters=args.iters,
        batch_size=args.batch_size)

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

    train_dataset = cfg.train_dataset
    if train_dataset is None:
        raise RuntimeError(
            'The training dataset is not specified in the configuration file.')
    elif len(train_dataset) == 0:
        raise ValueError(
            'The length of train_dataset is 0. Please check if your dataset is valid'
        )
    val_dataset = cfg.val_dataset if args.do_eval else None
    losses = cfg.loss
    # print(losses)
    # loss_hrnet = {
    #     'types': [
    #         {'type': 'CrossEntropyLoss'},
    #         {'type': 'PixelContrastCrossEntropyLoss', 'temperature': 0.1, 'base_temperature': 0.07, 'ignore_index': 255,
    #          'max_samples': 1024, 'max_views': 100}
    #     ],
    #     'coef': [1, 0.1]
    # }
    # loss_hrnet = {'types': [CrossEntropyLoss(), PixelContrastCrossEntropyLoss()], 'coef': [1, 0.1]}

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    config_check(cfg, train_dataset=train_dataset, val_dataset=val_dataset)

    # 第一次forward    train_server.train_singleData
    train_server.train(
        cfg.model,
        train_dataset,
        server1,
        server2,
        server3,
        val_dataset=val_dataset,
        optimizer=cfg.optimizer,
        save_dir=args.save_dir,
        iters=cfg.iters,
        batch_size=cfg.batch_size,
        resume_model=args.resume_model,
        save_interval=args.save_interval,
        log_iters=args.log_iters,
        num_workers=args.num_workers,
        use_vdl=args.use_vdl,
        losses=losses,
        # loss_hrnet=loss_hrnet,
        keep_checkpoint_max=args.keep_checkpoint_max,
        test_config=cfg.test_config,
        fp16=args.fp16,
        profiler_options=args.profiler_options)


def main_2(args, server1, server2, server3):
    if args.seed is not None:
        paddle.seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(
        args.cfg,
        learning_rate=args.learning_rate,
        iters=args.iters,
        batch_size=args.batch_size)

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

    train_dataset = cfg.train_dataset
    if train_dataset is None:
        raise RuntimeError(
            'The training dataset is not specified in the configuration file.')
    elif len(train_dataset) == 0:
        raise ValueError(
            'The length of train_dataset is 0. Please check if your dataset is valid'
        )
    val_dataset = cfg.val_dataset if args.do_eval else None
    losses = cfg.loss

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    config_check(cfg, train_dataset=train_dataset, val_dataset=val_dataset)

    # 第二次forward
    train_server.train_2(
        cfg.model,
        train_dataset,
        server1,
        server2,
        server3,
        val_dataset=val_dataset,
        optimizer=cfg.optimizer,
        save_dir=args.save_dir,
        iters=cfg.iters,
        batch_size=cfg.batch_size,
        resume_model=args.resume_model,
        save_interval=args.save_interval,
        log_iters=args.log_iters,
        num_workers=args.num_workers,
        use_vdl=args.use_vdl,
        losses=losses,
        keep_checkpoint_max=args.keep_checkpoint_max,
        test_config=cfg.test_config,
        fp16=args.fp16,
        profiler_options=args.profiler_options)


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

    # main(args, server1, server2, server3)
    logger.info("The subnetwork model is trained, and the server network is trained next.")
    main_2(args, server1, server2, server3)
