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
from paddleseg.core.val_server import evaluate, evaluate_client1, evaluate_client2, evaluate_client3
from paddleseg.models.dem_label import Slope_net, Aspect, Curvatures
from paddleseg.models.server1_Linear import server1_Linear
from paddleseg.models.server2_Linear import server2_Linear
from paddleseg.models.server3_Linear import server3_Linear

from paddleseg.models.losses.pixel_contrast_cross_entropy_loss import PixelContrastCrossEntropyLoss
from paddleseg.models.losses.cross_entropy_loss import CrossEntropyLoss

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
          server1,
          server2,
          server3,
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
          # loss_hrnet=None,
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

    linear1 = server1_Linear(num_classes=2)
    linear2 = server2_Linear(num_classes=2)
    linear3 = server3_Linear(num_classes=2)
    linear1.train()
    linear2.train()
    linear3.train()

    loss_hrnet = {'types': [CrossEntropyLoss(), PixelContrastCrossEntropyLoss()], 'coef': [1, 0.1]}

    # model.train()

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

    client1_avg_loss = 0.0
    client1_avg_loss_list = []
    client2_avg_loss = 0.0
    client2_avg_loss_list = []
    client3_avg_loss = 0.0
    client3_avg_loss_list = []

    iters_per_epoch = len(batch_sampler)
    best_mean_iou_1 = -1.0
    best_model_iter_1 = -1
    best_mean_iou_2 = -1.0
    best_model_iter_2 = -1
    best_mean_iou_3 = -1.0
    best_model_iter_3 = -1
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    save_models = deque()
    batch_start = time.time()
    best_F1_1 = -1.0
    best_F1_2 = -1.0
    best_F1_3 = -1.0

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
                # print("计算曲率坡度")
                slope_label = slope(dem)
                slope_label = (slope_label - paddle.min(slope_label)) / (
                        paddle.max(slope_label) - paddle.min(slope_label))
                aspect_label = aspect(dem)
                aspect_label = (aspect_label - paddle.min(aspect_label)) / (
                        paddle.max(aspect_label) - paddle.min(aspect_label))
                cal_label = cal(dem)
                cal_label = (cal_label - paddle.min(cal_label)) / (paddle.max(cal_label) - paddle.min(cal_label))
                # # cal_label2 = paddle.where(paddle.isnan(cal_label),0,cal_label)

                # print('s:',slope_label.shape) [2, 1, 1024, 1024]
                # print('a:',aspect_label.shape)    [2, 1, 1024, 1024]
                # print('c:',cal_label.shape)   [2, 1, 1024, 1024]

                recv_file(server1, "rev_xy_files1.tensor")
                # print("接收客户端1的特征图")
                recv_file(server2, "rev_xy_files2.tensor")
                # print("接收客户端2的特征图")
                # send_cmd(server2, 1)  # logits_list传送并已被接受
                # recv_file(server2, "rev_xy_files2_dem_pred.tensor")
                # print("接收客户端2的dem_pred的特征图")
                # send_cmd(server2, 2)  # dem_pred传送并已被接受
                recv_file(server3, "rev_xy_files3.tensor")
                # print("接收客户端3的特征图")

                opt = paddle.load("rev_xy_files1.tensor")
                dem = paddle.load("rev_xy_files2.tensor")
                # dem_pred = paddle.load("rev_xy_files2_dem_pred.tensor")
                sar = paddle.load("rev_xy_files3.tensor")

                opt = opt.detach()
                dem = dem.detach()
                # dem_pred = dem_pred.detach()
                sar = sar.detach()
                opt.stop_gradient = False
                dem.stop_gradient = False
                # dem_pred.stop_gradient = False
                sar.stop_gradient = False

                pred_client1 = linear1(opt)
                # print("pred_client1 = linear1(opt)")
                pred_client2, dem_pred = linear2(dem)
                # print("pred_client2, dem_pred = linear2(dem)")
                pred_client3 = linear3(sar)
                # print("pred_client3 = linear3(sar)")

                slope_loss = F.mse_loss(dem_pred, slope_label)
                aspect_loss = F.mse_loss(dem_pred, aspect_label)
                cur_loss = F.mse_loss(dem_pred, cal_label)

                # print(opt_logits_list.shape)
                client1_loss_list = loss_computation(
                    logits_list=pred_client1,
                    labels=labels,
                    losses=losses,
                    edges=edges)
                client1_loss = sum(client1_loss_list)
                client1_loss.backward()
                opt_grad = opt.grad
                paddle.save(opt_grad, "server_dfx_grad1.tensor")
                # print("发送客户端1的dfx")
                send_file(server1, "server_dfx_grad1.tensor")
                # time.sleep(10)

                client2_loss_list = loss_computation(
                    logits_list=pred_client2,
                    labels=labels,
                    losses=loss_hrnet,
                    edges=edges)
                client2_loss = sum(client2_loss_list) + slope_loss + aspect_loss + cur_loss
                client2_loss.backward()
                dem_grad = dem.grad
                # dem_pred_grad = dem_pred.grad
                paddle.save(dem_grad, "server_dfx_grad2.tensor")
                # print("发送客户端2的dfx")
                send_file(server2, "server_dfx_grad2.tensor")
                # recv_cmd(server2)
                # paddle.save(dem_pred_grad, "server_dfx_grad2_dem_pred.tensor")
                # send_file(server2, "server_dfx_grad2_dem_pred.tensor")
                # recv_cmd(server2)
                # time.sleep(10)

                # print(sar_logits_list.shape)
                client3_loss_list = loss_computation(
                    logits_list=pred_client3,
                    labels=labels,
                    losses=losses,
                    edges=edges)
                client3_loss = sum(client3_loss_list)
                client3_loss.backward()
                sar_grad = sar.grad
                paddle.save(sar_grad, "server_dfx_grad3.tensor")
                # print("发送客户端3的dfx")
                send_file(server3, "server_dfx_grad3.tensor")
                # time.sleep(10)

                # opt_grad = opt.grad
                # dem_grad = dem.grad
                # sar_grad = sar.grad

                # grad = fusion.grad
                # x_stop_gradient = x.detach()
                # x_stop_gradient.stop_gradient = True
                # if opt_grad is None:
                #     print("grad is none")
                #     print(opt_grad)

                # 第二次反向传播
                # recv_file(server1, "rev_xy_files1_fusion.tensor")
                # recv_file(server2, "rev_xy_files2_fusion.tensor")
                # recv_file(server3, "rev_xy_files3_fusion.tensor")
                # opt_fusion = paddle.load("rev_xy_files1_fusion.tensor")
                # dem_fusion = paddle.load("rev_xy_files2_fusion.tensor")
                # sar_fusion = paddle.load("rev_xy_files3_fusion.tensor")
                #
                # opt_fusion = opt_fusion.detach()
                # dem_fusion = dem_fusion.detach()
                # sar_fusion = sar_fusion.detach()
                # opt_fusion.stop_gradient = False
                # dem_fusion.stop_gradient = False
                # sar_fusion.stop_gradient = False
                # # print(fusion)
                # pred = ddp_model(opt_fusion, dem_fusion, sar_fusion) if nranks > 1 else model(opt_fusion, dem_fusion,
                #                                                                               sar_fusion)  #
                #
                # loss_list = loss_computation(
                #     logits_list=pred,
                #     labels=labels,
                #     losses=losses,
                #     edges=edges)
                # loss = sum(loss_list)
                # loss.backward()
                optimizer.step()
                # # loss_list = (client1_loss_list + client2_loss_list + client3_loss_list)/3
                loss_list = [(x + y + z) / 3 for x, y, z in
                             zip(client1_loss_list, client2_loss_list, client3_loss_list)]
                loss = (client1_loss + client2_loss + client3_loss) / 3
                # ?

            lr = optimizer.get_lr()

            # update lr
            if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                lr_sche = optimizer.user_defined_optimizer._learning_rate
            else:
                lr_sche = optimizer._learning_rate
            if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                lr_sche.step()

            train_profiler.add_profiler_step(profiler_options)
            linear1.clear_gradients()
            linear2.clear_gradients()
            linear3.clear_gradients()
            # model.clear_gradients()
            avg_loss += loss.numpy()[0]
            client1_avg_loss += client1_loss.numpy()[0]
            client2_avg_loss += client2_loss.numpy()[0]
            client3_avg_loss += client3_loss.numpy()[0]

            if not avg_loss_list:
                avg_loss_list = [l.numpy() for l in loss_list]
            else:
                for i in range(len(loss_list)):
                    avg_loss_list[i] += loss_list[i].numpy()

            if not client1_avg_loss_list:
                client1_avg_loss_list = [l.numpy() for l in client1_loss_list]
            else:
                for i in range(len(client1_loss_list)):
                    client1_avg_loss_list[i] += client1_loss_list[i].numpy()

            if not client2_avg_loss_list:
                client2_avg_loss_list = [l.numpy() for l in client2_loss_list]
            else:
                for i in range(len(client2_loss_list)):
                    client2_avg_loss_list[i] += client2_loss_list[i].numpy()

            if not client3_avg_loss_list:
                client3_avg_loss_list = [l.numpy() for l in client3_loss_list]
            else:
                for i in range(len(client3_loss_list)):
                    client3_avg_loss_list[i] += client3_loss_list[i].numpy()

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=batch_size)

            if (iter) % log_iters == 0 and local_rank == 0:
                avg_loss /= log_iters
                client1_avg_loss /= log_iters
                client2_avg_loss /= log_iters
                client3_avg_loss /= log_iters

                avg_loss_list = [l[0] / log_iters for l in avg_loss_list]
                client1_avg_loss_list = [l[0] / log_iters for l in client1_avg_loss_list]
                client2_avg_loss_list = [l[0] / log_iters for l in client2_avg_loss_list]
                client3_avg_loss_list = [l[0] / log_iters for l in client3_avg_loss_list]

                remain_iters = iters - iter
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)
                logger.info(
                    "[TRAIN] epoch: {}, iter: {}/{}, avg_loss: {:.4f}, Client1_loss: {:.4f}, Client2_loss: {:.4f}, Client3_loss: {:.4f}, lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
                        .format((iter - 1) // iters_per_epoch + 1, iter, iters,
                                avg_loss, client1_avg_loss, client2_avg_loss, client3_avg_loss, lr,
                                avg_train_batch_cost,
                                avg_train_reader_cost,
                                batch_cost_averager.get_ips_average(), eta))
                if use_vdl:
                    log_writer.add_scalar('Train/avg_loss', avg_loss, iter)
                    log_writer.add_scalar('Train/client1_loss', client1_avg_loss, iter)
                    log_writer.add_scalar('Train/client2_loss', client2_avg_loss, iter)
                    log_writer.add_scalar('Train/client3_loss', client3_avg_loss, iter)
                    # Record all losses if there are more than 2 losses.
                    if len(avg_loss_list) > 1:
                        avg_loss_dict = {}
                        for i, value in enumerate(avg_loss_list):
                            avg_loss_dict['loss_' + str(i)] = value
                        for key, value in avg_loss_dict.items():
                            log_tag = 'Train/' + key
                            log_writer.add_scalar(log_tag, value, iter)
                    if len(client1_avg_loss_list) > 1:
                        client1_avg_loss_dict = {}
                        for i, value in enumerate(client1_avg_loss_list):
                            client1_avg_loss_dict['client1_loss_' + str(i)] = value
                        for key, value in client1_avg_loss_dict.items():
                            log_tag = 'client1_Train/' + key
                            log_writer.add_scalar(log_tag, value, iter)
                    if len(client2_avg_loss_list) > 1:
                        client2_avg_loss_dict = {}
                        for i, value in enumerate(client2_avg_loss_list):
                            client2_avg_loss_dict['client2_loss_' + str(i)] = value
                        for key, value in client2_avg_loss_dict.items():
                            log_tag = 'client2_Train/' + key
                            log_writer.add_scalar(log_tag, value, iter)
                    if len(client3_avg_loss_list) > 1:
                        client3_avg_loss_dict = {}
                        for i, value in enumerate(client3_avg_loss_list):
                            client3_avg_loss_dict['client3_loss_' + str(i)] = value
                        for key, value in client3_avg_loss_dict.items():
                            log_tag = 'client3_Train/' + key
                            log_writer.add_scalar(log_tag, value, iter)

                    log_writer.add_scalar('Train/lr', lr, iter)
                    log_writer.add_scalar('Train/batch_cost',
                                          avg_train_batch_cost, iter)
                    log_writer.add_scalar('Train/reader_cost',
                                          avg_train_reader_cost, iter)

                avg_loss = 0.0
                avg_loss_list = []
                client1_avg_loss = 0.0
                client1_avg_loss_list = []
                client2_avg_loss = 0.0
                client2_avg_loss_list = []
                client3_avg_loss = 0.0
                client3_avg_loss_list = []
                reader_cost_averager.reset()
                batch_cost_averager.reset()

            if (iter % save_interval == 0
                or iter == iters) and (val_dataset is not None):
                num_workers = 1 if num_workers > 0 else 0

                if test_config is None:
                    test_config = {}

                # recv_file(server1, "rev_xy_files1_test.tensor")
                # fusion = paddle.load("rev_xy_files1_test.tensor")
                # mean_iou, acc, F1, _, _, _ = evaluate(
                #     model, val_dataset, server1, server2, server3, num_workers=num_workers, **test_config)
                mean_iou_1, acc_1, F1_1, _, _, _ = evaluate_client1(
                    linear1, val_dataset, server1, num_workers=num_workers, **test_config)
                mean_iou_2, acc_2, F1_2, _, _, _ = evaluate_client2(
                    linear2, val_dataset, server2, num_workers=num_workers, **test_config)
                mean_iou_3, acc_3, F1_3, _, _, _ = evaluate_client3(
                    linear3, val_dataset, server3, num_workers=num_workers, **test_config)

                linear1.train()
                linear2.train()
                linear3.train()
                # model.train()

            if (iter % save_interval == 0 or iter == iters) and local_rank == 0:
                # current_save_dir = os.path.join(save_dir,
                #                                 "iter_{}".format(iter))
                # if not os.path.isdir(current_save_dir):
                #     os.makedirs(current_save_dir)
                # paddle.save(model.state_dict(),
                #             os.path.join(current_save_dir, 'model.pdparams'))
                # paddle.save(optimizer.state_dict(),
                #             os.path.join(current_save_dir, 'model.pdopt'))
                # save_models.append(current_save_dir)
                # if len(save_models) > keep_checkpoint_max > 0:
                #     model_to_remove = save_models.popleft()
                #     shutil.rmtree(model_to_remove)

                if val_dataset is not None:
                    if mean_iou_1 > best_mean_iou_1 and F1_1[1] > best_F1_1:
                        best_mean_iou_1 = mean_iou_1
                        best_F1_1 = F1_1[1]
                        best_model_iter_1 = iter
                        send_cmd(server1, best_model_iter_1)
                        # send_cmd(server2, best_model_iter)
                        # send_cmd(server3, best_model_iter)
                    else:
                        send_cmd(server1, '-1')
                        # send_cmd(server2, '-1')
                        # send_cmd(server3, '-1')
                    if mean_iou_2 > best_mean_iou_2 and F1_2[1] > best_F1_2:
                        best_mean_iou_2 = mean_iou_2
                        best_F1_2 = F1_2[1]
                        best_model_iter_2 = iter
                        send_cmd(server2, best_model_iter_2)
                        # send_cmd(server2, best_model_iter)
                        # send_cmd(server3, best_model_iter)
                    else:
                        send_cmd(server2, '-1')
                        # send_cmd(server2, '-1')
                        # send_cmd(server3, '-1')
                    if mean_iou_3 > best_mean_iou_3 and F1_3[1] > best_F1_3:
                        best_mean_iou_3 = mean_iou_3
                        best_F1_3 = F1_3[1]
                        best_model_iter_3 = iter
                        send_cmd(server3, best_model_iter_3)
                        # send_cmd(server2, best_model_iter)
                        # send_cmd(server3, best_model_iter)
                    else:
                        send_cmd(server3, '-1')
                        # send_cmd(server2, '-1')
                        # send_cmd(server3, '-1')

                    logger.info(
                        '[EVAL] Client1 : The model with the best validation mIoU ({:.4f}) and best F1 ({:.4f}) was saved at iter {}.'
                            .format(best_mean_iou_1, best_F1_1, best_model_iter_1))
                    logger.info(
                        '[EVAL] Client2 : The model with the best validation mIoU ({:.4f}) and best F1 ({:.4f}) was saved at iter {}.'
                            .format(best_mean_iou_2, best_F1_2, best_model_iter_2))
                    logger.info(
                        '[EVAL] Client3 : The model with the best validation mIoU ({:.4f}) and best F1 ({:.4f}) was saved at iter {}.'
                            .format(best_mean_iou_3, best_F1_3, best_model_iter_3))

                    if use_vdl:
                        log_writer.add_scalar('Evaluate/mIoU', mean_iou, iter)
                        log_writer.add_scalar('Evaluate/Acc', acc, iter)
            batch_start = time.time()

    # Calculate flops.
    # if local_rank == 0:
    #     _, c, h, w = images.shape
    #     _ = paddle.flops(
    #         model, [1, c, h, w],
    #         custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})

    # Sleep for half a second to let dataloader release resources.
    time.sleep(0.5)
    if use_vdl:
        log_writer.close()


def train_2(model,
            train_dataset,
            server1,
            server2,
            server3,
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

    # linear1 = server1_Linear(num_classes=2)
    # linear2 = server2_Linear(num_classes=2)
    # linear3 = server3_Linear(num_classes=2)
    # linear1.train()
    # linear2.train()
    # linear3.train()
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

                # recv_file(server1, "rev_xy_files1.tensor")
                # recv_file(server2, "rev_xy_files2.tensor")
                # recv_file(server3, "rev_xy_files3.tensor")
                # opt = paddle.load("rev_xy_files1.tensor")
                # dem = paddle.load("rev_xy_files2.tensor")
                # sar = paddle.load("rev_xy_files3.tensor")
                #
                # opt = opt.detach()
                # dem = dem.detach()
                # sar = sar.detach()
                # opt.stop_gradient = False
                # dem.stop_gradient = False
                # sar.stop_gradient = False
                #
                # pred_client1 = linear1(opt)
                # pred_client2 = linear2(dem)
                # pred_client3 = linear3(sar)
                #
                # client1_loss_list = loss_computation(
                #     logits_list=pred_client1,
                #     labels=labels,
                #     losses=losses,
                #     edges=edges)
                # client1_loss = sum(client1_loss_list)
                # client1_loss.backward()
                #
                # client2_loss_list = loss_computation(
                #     logits_list=pred_client2,
                #     labels=labels,
                #     losses=losses,
                #     edges=edges)
                # client2_loss = sum(client2_loss_list)
                # client2_loss.backward()
                #
                # client3_loss_list = loss_computation(
                #     logits_list=pred_client3,
                #     labels=labels,
                #     losses=losses,
                #     edges=edges)
                # client3_loss = sum(client3_loss_list)
                # client3_loss.backward()
                #
                # opt_grad = opt.grad
                # dem_grad = dem.grad
                # sar_grad = sar.grad
                # # grad = fusion.grad
                # # x_stop_gradient = x.detach()
                # # x_stop_gradient.stop_gradient = True
                # if opt_grad is None:
                #     print("grad is none")
                #     print(opt_grad)
                # paddle.save(opt_grad, "server_dfx_grad1.tensor")
                # paddle.save(dem_grad, "server_dfx_grad2.tensor")
                # paddle.save(sar_grad, "server_dfx_grad3.tensor")
                # send_file(server1, "server_dfx_grad1.tensor")
                # send_file(server2, "server_dfx_grad2.tensor")
                # send_file(server3, "server_dfx_grad3.tensor")

                # 第二次反向传播
                recv_file(server1, "rev_xy_files1_fusion.tensor")
                send_cmd(server1, '1')
                recv_file(server2, "rev_xy_files2_fusion.tensor")
                send_cmd(server2, '1')
                recv_file(server3, "rev_xy_files3_fusion.tensor")
                send_cmd(server3, '1')
                opt_fusion = paddle.load("rev_xy_files1_fusion.tensor")
                dem_fusion = paddle.load("rev_xy_files2_fusion.tensor")
                hill_fusion = paddle.load("rev_xy_files3_fusion.tensor")

                opt_fusion = opt_fusion.detach()
                dem_fusion = dem_fusion.detach()
                hill_fusion = hill_fusion.detach()
                opt_fusion.stop_gradient = False
                dem_fusion.stop_gradient = False
                hill_fusion.stop_gradient = False
                # print(fusion)
                pred, dem_pred = ddp_model(opt_fusion, dem_fusion, hill_fusion) if nranks > 1 else model(opt_fusion,
                                                                                                        dem_fusion,
                                                                                                        hill_fusion)  #

                slope_label = slope(dem)
                slope_label = (slope_label - paddle.min(slope_label)) / (
                        paddle.max(slope_label) - paddle.min(slope_label))
                aspect_label = aspect(dem)
                aspect_label = (aspect_label - paddle.min(aspect_label)) / (
                        paddle.max(aspect_label) - paddle.min(aspect_label))
                cal_label = cal(dem)
                cal_label = (cal_label - paddle.min(cal_label)) / (paddle.max(cal_label) - paddle.min(cal_label))
                # cal_label2 = paddle.where(paddle.isnan(cal_label),0,cal_label)

                slope_loss = F.mse_loss(dem_pred, slope_label)
                aspect_loss = F.mse_loss(dem_pred, aspect_label)
                cur_loss = F.mse_loss(dem_pred, cal_label)

                loss_list = loss_computation(
                    logits_list=pred,
                    labels=labels,
                    losses=losses,
                    edges=edges)
                loss = sum(loss_list) + slope_loss + aspect_loss + cur_loss
                loss.backward()
                optimizer.step()

                # ?

            lr = optimizer.get_lr()

            # update lr
            if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                lr_sche = optimizer.user_defined_optimizer._learning_rate
            else:
                lr_sche = optimizer._learning_rate
            if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                lr_sche.step()

            train_profiler.add_profiler_step(profiler_options)
            # linear1.clear_gradients()
            # linear2.clear_gradients()
            # linear3.clear_gradients()
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

                # linear1.train()
                # linear2.train()
                # linear3.train()
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

                        # send_cmd(server1, best_model_iter)
                        # send_cmd(server2, best_model_iter)
                        # send_cmd(server3, best_model_iter)

                        best_model_dir = os.path.join(save_dir, "best_model")
                        # best_model_dir2 = os.path.join(save_dir, "best_model")
                        paddle.save(
                            model.state_dict(),
                            os.path.join(best_model_dir, 'model.pdparams'))
                    # else:
                    #     send_cmd(server1, '-1')
                    #     send_cmd(server2, '-1')
                    #     send_cmd(server3, '-1')

                    logger.info(
                        '[EVAL] The model with the best validation mIoU ({:.4f}) and best F1 ({:.4f}) was saved at iter {}.'
                            .format(best_mean_iou, best_F1, best_model_iter))

                    if use_vdl:
                        log_writer.add_scalar('Evaluate/mIoU', mean_iou, iter)
                        log_writer.add_scalar('Evaluate/Acc', acc, iter)
            batch_start = time.time()

    # Calculate flops.
    # if local_rank == 0:
    #     _, c, h, w = images.shape
    #     _ = paddle.flops(
    #         model, [1, c, h, w],
    #         custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})

    # Sleep for half a second to let dataloader release resources.
    time.sleep(0.5)
    if use_vdl:
        log_writer.close()


def train_singleData(model,
                     train_dataset,
                     server1,
                     server2,
                     server3,
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
    # model.train()
    linear1 = server1_Linear(num_classes=2)
    linear1.train()

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

                # logits_list, dem_pred = ddp_model(images, opt, dem_normalize, sar_normalize) if nranks > 1 else model(
                #     images, opt, dem_normalize, sar_normalize)  #
                # logits_list = ddp_model(images, opt, dem_normalize, sar_normalize) if nranks > 1 else model(
                #     images, opt, dem_normalize, sar_normalize)  #

                # segformer opt
                recv_file(server1, "rev_xy_files1.tensor")
                opt = paddle.load("rev_xy_files1.tensor")
                opt = opt.detach()
                opt.stop_gradient = False
                logits_list = linear1(opt)

                loss_list = loss_computation(
                    logits_list=logits_list,
                    labels=labels,
                    losses=losses,
                    edges=edges)
                loss = sum(loss_list)
                loss.backward()

                opt_grad = opt.grad
                paddle.save(opt_grad, "server_dfx_grad1.tensor")
                send_file(server1, "server_dfx_grad1.tensor")

                # recv_file(server1, "logits_list_rev_xy_files1.tensor")
                # logits_list = paddle.load("logits_list_rev_xy_files1.tensor")
                # print(logits_list)
                # logits_list[0].stop_gradient = False
                # logits_list.stop_gradient = False

                # 将 logits_list 中每个 tensor 的 stop_gradient 设置为 False
                # for i in range(len(logits_list)):
                #     logits_list[i].stop_gradient = False

                # slope_label = slope(dem)
                # slope_label = (slope_label - paddle.min(slope_label)) / (
                #         paddle.max(slope_label) - paddle.min(slope_label))
                # aspect_label = aspect(dem)
                # aspect_label = (aspect_label - paddle.min(aspect_label)) / (
                #         paddle.max(aspect_label) - paddle.min(aspect_label))
                # cal_label = cal(dem)
                # cal_label = (cal_label - paddle.min(cal_label)) / (paddle.max(cal_label) - paddle.min(cal_label))
                # # cal_label2 = paddle.where(paddle.isnan(cal_label),0,cal_label)
                #
                # slope_loss = F.mse_loss(dem_pred, slope_label)
                # aspect_loss = F.mse_loss(dem_pred, aspect_label)
                # cur_loss = F.mse_loss(dem_pred, cal_label)

                # print(slope_loss)
                # print(cal_label)
                # print(cur_loss)

                # print(loss)

                # paddle.save(labels, "loss_server_dfx_grad1.tensor")
                # send_file(server1, "loss_server_dfx_grad1.tensor")

                # loss.backward()
                optimizer.step()

                # print(model.parameters())

            lr = optimizer.get_lr()

            # update lr
            if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                lr_sche = optimizer.user_defined_optimizer._learning_rate
            else:
                lr_sche = optimizer._learning_rate
            if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                lr_sche.step()

            train_profiler.add_profiler_step(profiler_options)

            # model.clear_gradients()
            linear1.clear_gradients()

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

                mean_iou, acc, F1, _, _, _ = evaluate_client1(
                    linear1, val_dataset, server1, num_workers=num_workers, **test_config)

                linear1.train()

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
                        best_model_dir = os.path.join(save_dir, "best_model")
                        # best_model_dir2 = os.path.join(save_dir, "best_model")
                        paddle.save(
                            model.state_dict(),
                            os.path.join(best_model_dir, 'model.pdparams'))
                    else:
                        send_cmd(server1, '-1')

                    logger.info(
                        '[EVAL] The model with the best validation mIoU ({:.4f}) and best F1 ({:.4f}) was saved at iter {}.'
                            .format(best_mean_iou, best_F1, best_model_iter))

                    if use_vdl:
                        log_writer.add_scalar('Evaluate/mIoU', mean_iou, iter)
                        log_writer.add_scalar('Evaluate/Acc', acc, iter)
            batch_start = time.time()

    # # Calculate flops.
    # if local_rank == 0:
    #     _, c, h, w = images.shape
    #     _ = paddle.flops(
    #         model, [1, c, h, w],
    #         custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})

    # Sleep for half a second to let dataloader release resources.
    time.sleep(0.5)
    if use_vdl:
        log_writer.close()
