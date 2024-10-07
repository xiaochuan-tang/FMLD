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

import os
import time
from collections import deque
import shutil

import paddle
import paddle.nn.functional as F

from paddleseg.utils import (TimeAverager, calculate_eta, resume, logger,
                             worker_init_fn, train_profiler, op_flops_funs)
from paddleseg.core.val_server import evaluate
from paddleseg.models.dem_label import Slope_net, Aspect, Curvatures

import socket
import numpy
# from paddleseg.core.udprely import *

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


def check_logits_losses(logits_list, losses):
    len_logits = len(logits_list)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits_list should equal to the types of loss config: {} != {}.'
                .format(len_logits, len_losses))


def loss_computation(logits_list, labels, losses, edges=None):
    check_logits_losses(logits_list, losses)
    loss_list = []
    for i in range(len(logits_list)):
        logits = logits_list[i]
        loss_i = losses['types'][i]
        # Whether to use edges as labels According to loss type.
        if loss_i.__class__.__name__ in ('BCELoss',
                                         'FocalLoss') and loss_i.edge_label:
            loss_list.append(losses['coef'][i] * loss_i(logits, edges))
        elif loss_i.__class__.__name__ in ("KLLoss",):
            loss_list.append(losses['coef'][i] * loss_i(
                logits_list[0], logits_list[1].detach()))
        else:
            loss_list.append(losses['coef'][i] * loss_i(logits, labels))
    return loss_list


def train(model,
          train_dataset,
          val_dataset=None,
          optimizer=None,
          save_dir='output',
          iters=10000,
          batch_size=2,
          resume_model=None,
          save_interval=1000,
          log_iters=10,
          num_workers=0,
          use_vdl=False,
          losses=None,
          keep_checkpoint_max=5,
          test_config=None,
          fp16=False,
          profiler_options=None):
    """
    Launch training.

    Args:
        model（nn.Layer): A sementic segmentation model.
        train_dataset (paddle.io.Dataset): Used to read and process training datasets.
        val_dataset (paddle.io.Dataset, optional): Used to read and process validation datasets.
        optimizer (paddle.optimizer.Optimizer): The optimizer.
        save_dir (str, optional): The directory for saving the model snapshot. Default: 'output'.
        iters (int, optional): How may iters to train the model. Defualt: 10000.
        batch_size (int, optional): Mini batch size of one gpu or cpu. Default: 2.
        resume_model (str, optional): The path of resume model.
        save_interval (int, optional): How many iters to save a model snapshot once during training. Default: 1000.
        log_iters (int, optional): Display logging information at every log_iters. Default: 10.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        use_vdl (bool, optional): Whether to record the data to VisualDL during training. Default: False.
        losses (dict, optional): A dict including 'types' and 'coef'. The length of coef should equal to 1 or len(losses['types']).
            The 'types' item is a list of object of paddleseg.models.losses while the 'coef' item is a list of the relevant coefficient.
        keep_checkpoint_max (int, optional): Maximum number of checkpoints to save. Default: 5.
        test_config(dict, optional): Evaluation config.
        fp16 (bool, optional): Whether to use amp.
        profiler_options (str, optional): The option of train profiler.
    """

    ########################### init socket###################
    # 基于 Socket 的服务器端程序，用于接收和发送文件数据
    # 在服务器端创建一个套接字，将服务器端套接字绑定到指定的地址和端口
    # socket.AF_INET:表示使用IPv4 地址族;socket.SOCK_STREAM:TCP 协议
    sock1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 通过调用 bind() 方法，服务器套接字与指定的地址和端口进行绑定，以便等待客户端的连接。
    sock1.bind(('172.26.101.204', 8001))

    sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock2.bind(('172.26.101.204', 10000))

    sock3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock3.bind(('172.26.101.204', 10001))

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

    model.train()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank

    start_iter = 0
    if resume_model is not None:
        start_iter = resume(model, optimizer, resume_model)

    if not os.path.isdir(save_dir):
        if os.path.exists(save_dir):
            os.remove(save_dir)
        os.makedirs(save_dir)

    if nranks > 1:
        paddle.distributed.fleet.init(is_collective=True)
        optimizer = paddle.distributed.fleet.distributed_optimizer(
            optimizer)  # The return is Fleet object
        ddp_model = paddle.distributed.fleet.distributed_model(model)

    batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
        worker_init_fn=worker_init_fn,
    )

    # use amp
    if fp16:
        logger.info('use amp to train')
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    if use_vdl:
        from visualdl import LogWriter
        log_writer = LogWriter(save_dir)

    avg_loss = 0.0
    avg_loss_list = []
    iters_per_epoch = len(batch_sampler)
    best_mean_iou = -1.0
    best_model_iter = -1
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    save_models = deque()
    batch_start = time.time()
    best_F1 = -1.0

    slope_loss = aspect_loss = cur_loss = 0.0
    slope = Slope_net()
    aspect = Aspect()
    cal = Curvatures()

    iter = start_iter
    while iter < iters:
        for data in loader:
            iter += 1
            if iter > iters:
                version = paddle.__version__
                if version == '2.1.2':
                    continue
                else:
                    break
            reader_cost_averager.record(time.time() - batch_start)
            images = data[0]
            opt = data[1]
            dem = data[2]
            sar = data[3]
            labels = data[4].astype('int64')
            edges = None
            if len(data) == 3:
                edges = data[2].astype('int64')
            if hasattr(model, 'data_format') and model.data_format == 'NHWC':
                images = images.transpose((0, 2, 3, 1))

            if fp16:
                with paddle.amp.auto_cast(
                        enable=True,
                        custom_white_list={
                            "elementwise_add", "batch_norm", "sync_batch_norm"
                        },
                        custom_black_list={'bilinear_interp_v2'}):
                    if nranks > 1:
                        logits_list = ddp_model(images)
                    else:
                        logits_list = model(images)
                    loss_list = loss_computation(
                        logits_list=logits_list,
                        labels=labels,
                        losses=losses,
                        edges=edges)
                    loss = sum(loss_list)

                scaled = scaler.scale(loss)  # scale the loss
                scaled.backward()  # do backward
                if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                    scaler.minimize(optimizer.user_defined_optimizer, scaled)
                else:
                    scaler.minimize(optimizer, scaled)  # update parameters
            else:

                # dem_normalize = (dem - paddle.min(dem)) / (paddle.max(dem) - paddle.min(dem) + 1e-6)  # dem归一化 输入网络训练
                # sar_normalize = (sar - paddle.min(sar)) / (paddle.max(sar) - paddle.min(sar) + 1e-6)  # sar归一化 输入网络训练

                recv_file(server1, "rev_xy_files1.tensor")
                recv_file(server2, "rev_xy_files2.tensor")
                recv_file(server3, "rev_xy_files3.tensor")
                opt = paddle.load("rev_xy_files1.tensor")
                dem = paddle.load("rev_xy_files2.tensor")
                sar = paddle.load("rev_xy_files3.tensor")

                opt = opt.detach()
                dem = dem.detach()
                sar = sar.detach()
                opt.stop_gradient = False
                dem.stop_gradient = False
                sar.stop_gradient = False

                # print(fusion)
                pred = ddp_model(opt, dem, sar) if nranks > 1 else model(opt, dem, sar)  #

                loss_list = loss_computation(
                    logits_list=pred,
                    labels=labels,
                    losses=losses,
                    edges=edges)
                loss = sum(loss_list)
                loss.backward()
                optimizer.step()

                # ?

                opt_grad = opt.grad
                dem_grad = dem.grad
                sar_grad = sar.grad
                # grad = fusion.grad
                # x_stop_gradient = x.detach()
                # x_stop_gradient.stop_gradient = True
                if opt_grad is None:
                    print("grad is none")
                    print(opt_grad)
                paddle.save(opt_grad, "server_dfx_grad1.tensor")
                paddle.save(dem_grad, "server_dfx_grad2.tensor")
                paddle.save(sar_grad, "server_dfx_grad3.tensor")
                send_file(server1, "server_dfx_grad1.tensor")
                send_file(server2, "server_dfx_grad2.tensor")
                send_file(server3, "server_dfx_grad3.tensor")

            lr = optimizer.get_lr()

            # update lr
            if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                lr_sche = optimizer.user_defined_optimizer._learning_rate
            else:
                lr_sche = optimizer._learning_rate
            if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                lr_sche.step()

            train_profiler.add_profiler_step(profiler_options)

            model.clear_gradients()
            avg_loss += loss.numpy()[0]
            if not avg_loss_list:
                avg_loss_list = [l.numpy() for l in loss_list]
            else:
                for i in range(len(loss_list)):
                    avg_loss_list[i] += loss_list[i].numpy()
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=batch_size)

            if (iter) % log_iters == 0 and local_rank == 0:
                avg_loss /= log_iters
                avg_loss_list = [l[0] / log_iters for l in avg_loss_list]
                remain_iters = iters - iter
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)
                logger.info(
                    "[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
                        .format((iter - 1) // iters_per_epoch + 1, iter, iters,
                                avg_loss, lr, avg_train_batch_cost,
                                avg_train_reader_cost,
                                batch_cost_averager.get_ips_average(), eta))
                if use_vdl:
                    log_writer.add_scalar('Train/loss', avg_loss, iter)
                    # Record all losses if there are more than 2 losses.
                    if len(avg_loss_list) > 1:
                        avg_loss_dict = {}
                        for i, value in enumerate(avg_loss_list):
                            avg_loss_dict['loss_' + str(i)] = value
                        for key, value in avg_loss_dict.items():
                            log_tag = 'Train/' + key
                            log_writer.add_scalar(log_tag, value, iter)

                    log_writer.add_scalar('Train/lr', lr, iter)
                    log_writer.add_scalar('Train/batch_cost',
                                          avg_train_batch_cost, iter)
                    log_writer.add_scalar('Train/reader_cost',
                                          avg_train_reader_cost, iter)
                avg_loss = 0.0
                avg_loss_list = []
                reader_cost_averager.reset()
                batch_cost_averager.reset()

            if (iter % save_interval == 0
                or iter == iters) and (val_dataset is not None):
                num_workers = 1 if num_workers > 0 else 0

                if test_config is None:
                    test_config = {}

                # recv_file(server1, "rev_xy_files1_test.tensor")
                # fusion = paddle.load("rev_xy_files1_test.tensor")
                mean_iou, acc, F1, _, _, _ = evaluate(
                    model, val_dataset, server1, server2, server3, num_workers=num_workers, **test_config)

                model.train()

            if (iter % save_interval == 0 or iter == iters) and local_rank == 0:
                current_save_dir = os.path.join(save_dir,
                                                "iter_{}".format(iter))
                if not os.path.isdir(current_save_dir):
                    os.makedirs(current_save_dir)
                paddle.save(model.state_dict(),
                            os.path.join(current_save_dir, 'model.pdparams'))
                paddle.save(optimizer.state_dict(),
                            os.path.join(current_save_dir, 'model.pdopt'))
                save_models.append(current_save_dir)
                if len(save_models) > keep_checkpoint_max > 0:
                    model_to_remove = save_models.popleft()
                    shutil.rmtree(model_to_remove)

                if val_dataset is not None:
                    if mean_iou > best_mean_iou and F1[1] > best_F1:
                        best_mean_iou = mean_iou
                        best_F1 = F1[1]
                        best_model_iter = iter

                        send_cmd(server1, best_model_iter)
                        send_cmd(server2, best_model_iter)
                        send_cmd(server3, best_model_iter)

                        best_model_dir = os.path.join(save_dir, "best_model")
                        # best_model_dir2 = os.path.join(save_dir, "best_model")
                        paddle.save(
                            model.state_dict(),
                            os.path.join(best_model_dir, 'model.pdparams'))
                    else:
                        send_cmd(server1, '-1')
                        send_cmd(server2, '-1')
                        send_cmd(server3, '-1')

                    logger.info(
                        '[EVAL] The model with the best validation mIoU ({:.4f}) and best F1 ({:.4f}) was saved at iter {}.'
                            .format(best_mean_iou, best_F1, best_model_iter))

                    if use_vdl:
                        log_writer.add_scalar('Evaluate/mIoU', mean_iou, iter)
                        log_writer.add_scalar('Evaluate/Acc', acc, iter)
            batch_start = time.time()

    # Calculate flops.
    if local_rank == 0:
        _, c, h, w = images.shape
        _ = paddle.flops(
            model, [1, c, h, w],
            custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})

    # Sleep for half a second to let dataloader release resources.
    time.sleep(0.5)
    if use_vdl:
        log_writer.close()
