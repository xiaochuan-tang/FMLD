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

import numpy as np
import time
import paddle
import paddle.nn.functional as F

from paddleseg.utils import metrics, TimeAverager, calculate_eta, logger, progbar
from paddleseg.core import infer_server

np.set_printoptions(suppress=True)


def recv_cmd(sock_fd):
    cmd = sock_fd.recv(128).decode()
    return cmd


def send_cmd(sock_fd, cmd):
    cmd = str(cmd)
    sock_fd.send(cmd.encode())


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


def evaluate(model,
             eval_dataset,
             server1,
             server2,
             server3,
             aug_eval=False,
             scales=1.0,
             flip_horizontal=True,
             flip_vertical=False,
             is_slide=False,
             stride=None,
             crop_size=None,
             num_workers=0,
             print_detail=True):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A sementic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        aug_eval (bool, optional): Whether to use mulit-scales and flip augment for evaluation. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_eval` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_eval` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_eval` is True. Default: False.
        is_slide (bool, optional): Whether to evaluate by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.

    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    model.eval()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
        ):
            paddle.distributed.init_parallel_env()
    batch_sampler = paddle.io.DistributedBatchSampler(
        eval_dataset, batch_size=1, shuffle=False, drop_last=False)
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
    )

    total_iters = len(loader)
    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0

    if print_detail:
        logger.info(
            "Start evaluating (total_samples: {}, total_iters: {})...".format(
                len(eval_dataset), total_iters))
    # TODO(chenguowei): fix log print error with multi-gpus
    progbar_val = progbar.Progbar(
        target=total_iters, verbose=1 if nranks < 2 else 2)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    with paddle.no_grad():
        for iter, (im, opt, dem, sar, label) in enumerate(loader):
            reader_cost_averager.record(time.time() - batch_start)
            label = label.astype('int64')
            dem_normalize = (dem - paddle.min(dem)) / (paddle.max(dem) - paddle.min(dem))
            sar_normalize = (sar - paddle.min(sar)) / (paddle.max(sar) - paddle.min(sar))
            ori_shape = label.shape[-2:]
            if aug_eval:
                pred = infer_server.aug_inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=eval_dataset.transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                recv_file(server1, "rev_xy_files1_test_eva.tensor")
                send_cmd(server1, '2')
                recv_file(server2, "rev_xy_files2_test_eva.tensor")
                send_cmd(server2, '2')
                recv_file(server3, "rev_xy_files3_test_eva.tensor")
                send_cmd(server3, '2')
                opt = paddle.load("rev_xy_files1_test_eva.tensor")
                dem = paddle.load("rev_xy_files2_test_eva.tensor")
                hill = paddle.load("rev_xy_files3_test_eva.tensor")
                # print(fusion.shape)
                pred = infer_server.inference(
                    model,
                    opt,
                    dem,
                    hill,
                    ori_shape=ori_shape,
                    transforms=eval_dataset.transforms.transforms,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)

            intersect_area, pred_area, label_area = metrics.calculate_area(
                pred,
                label,
                eval_dataset.num_classes,
                ignore_index=eval_dataset.ignore_index)

            # Gather from all ranks
            if nranks > 1:
                intersect_area_list = []
                pred_area_list = []
                label_area_list = []
                paddle.distributed.all_gather(intersect_area_list,
                                              intersect_area)
                paddle.distributed.all_gather(pred_area_list, pred_area)
                paddle.distributed.all_gather(label_area_list, label_area)

                # Some image has been evaluated and should be eliminated in last iter
                if (iter + 1) * nranks > len(eval_dataset):
                    valid = len(eval_dataset) - iter * nranks
                    intersect_area_list = intersect_area_list[:valid]
                    pred_area_list = pred_area_list[:valid]
                    label_area_list = label_area_list[:valid]

                for i in range(len(intersect_area_list)):
                    intersect_area_all = intersect_area_all + intersect_area_list[
                        i]
                    pred_area_all = pred_area_all + pred_area_list[i]
                    label_area_all = label_area_all + label_area_list[i]
            else:
                intersect_area_all = intersect_area_all + intersect_area
                pred_area_all = pred_area_all + pred_area
                label_area_all = label_area_all + label_area
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            if local_rank == 0 and print_detail:
                progbar_val.update(iter + 1, [('batch_cost', batch_cost),
                                              ('reader cost', reader_cost)])
            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

    class_iou, miou = metrics.mean_iou(intersect_area_all, pred_area_all,
                                       label_area_all)
    class_acc, acc = metrics.precision(intersect_area_all, pred_area_all)
    _, class_precision, class_rec = metrics.class_measurement(intersect_area_all, pred_area_all, label_area_all)
    kappa, po = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)
    # class_rec,mrec = metrics.recall(intersect_area_all,label_area_all)
    F1 = 2 * class_precision * class_rec / (class_precision + class_rec)
    meanF1 = np.mean(F1)
    if print_detail:
        infor = "[EVAL] #Images: {} mIoU: {:.4f} Acc: {:.4f} Pre:{:.4f} Rec:{:.4f} F1:{:.4f} Kappa: {:.4f},poacc:{:.4f}".format(
            len(eval_dataset), miou, acc, np.mean(class_precision), np.mean(class_rec), meanF1, kappa, po)
        logger.info(infor)
        logger.info("[EVAL] Class IoU: \n" + str(np.round(class_iou, 4)))
        # logger.info("[EVAL] Class Acc: \n" + str(np.round(class_acc, 4)))
        logger.info("[EVAL] Class Pre: \n" + str(np.round(class_precision, 4)))
        logger.info("[EVAL] Class Rec: \n" + str(np.round(class_rec, 4)))
        logger.info("[EVAL] Class F1: \n" + str(np.round(F1, 4)))
    return miou, acc, F1, class_iou, class_acc, kappa


# 下面的函数主要是用于联邦学习第一次训练子网络模型，加载的模型为服务器中的linear网络

def evaluate_client1(model,
                     eval_dataset,
                     server1,
                     aug_eval=False,
                     scales=1.0,
                     flip_horizontal=True,
                     flip_vertical=False,
                     is_slide=False,
                     stride=None,
                     crop_size=None,
                     num_workers=0,
                     print_detail=True):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A sementic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        aug_eval (bool, optional): Whether to use mulit-scales and flip augment for evaluation. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_eval` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_eval` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_eval` is True. Default: False.
        is_slide (bool, optional): Whether to evaluate by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.

    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    model.eval()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
        ):
            paddle.distributed.init_parallel_env()
    batch_sampler = paddle.io.DistributedBatchSampler(
        eval_dataset, batch_size=1, shuffle=False, drop_last=False)
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
    )

    total_iters = len(loader)
    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0

    if print_detail:
        logger.info(
            "Start evaluating (total_samples: {}, total_iters: {})...".format(
                len(eval_dataset), total_iters))
    # TODO(chenguowei): fix log print error with multi-gpus
    progbar_val = progbar.Progbar(
        target=total_iters, verbose=1 if nranks < 2 else 2)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    with paddle.no_grad():
        for iter, (im, opt, dem, sar, label) in enumerate(loader):
            reader_cost_averager.record(time.time() - batch_start)
            label = label.astype('int64')
            dem_normalize = (dem - paddle.min(dem)) / (paddle.max(dem) - paddle.min(dem))
            sar_normalize = (sar - paddle.min(sar)) / (paddle.max(sar) - paddle.min(sar))
            ori_shape = label.shape[-2:]
            if aug_eval:
                pred = infer_server.aug_inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=eval_dataset.transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                recv_file(server1, "rev_xy_files1_test.tensor")
                # recv_file(server2, "rev_xy_files2_test.tensor")
                # recv_file(server3, "rev_xy_files3_test.tensor")
                opt = paddle.load("rev_xy_files1_test.tensor")

                # dem = paddle.load("rev_xy_files2_test.tensor")
                # sar = paddle.load("rev_xy_files3_test.tensor")
                # print(fusion.shape)

                pred = infer_server.inference_client1(
                    model,
                    opt,
                    ori_shape=ori_shape,
                    transforms=eval_dataset.transforms.transforms,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            # print(pred.shape)
            # print(label.shape)

            intersect_area, pred_area, label_area = metrics.calculate_area(
                pred,
                label,
                eval_dataset.num_classes,
                ignore_index=eval_dataset.ignore_index)

            # Gather from all ranks
            if nranks > 1:
                intersect_area_list = []
                pred_area_list = []
                label_area_list = []
                paddle.distributed.all_gather(intersect_area_list,
                                              intersect_area)
                paddle.distributed.all_gather(pred_area_list, pred_area)
                paddle.distributed.all_gather(label_area_list, label_area)

                # Some image has been evaluated and should be eliminated in last iter
                if (iter + 1) * nranks > len(eval_dataset):
                    valid = len(eval_dataset) - iter * nranks
                    intersect_area_list = intersect_area_list[:valid]
                    pred_area_list = pred_area_list[:valid]
                    label_area_list = label_area_list[:valid]

                for i in range(len(intersect_area_list)):
                    intersect_area_all = intersect_area_all + intersect_area_list[
                        i]
                    pred_area_all = pred_area_all + pred_area_list[i]
                    label_area_all = label_area_all + label_area_list[i]
            else:
                intersect_area_all = intersect_area_all + intersect_area
                pred_area_all = pred_area_all + pred_area
                label_area_all = label_area_all + label_area
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            if local_rank == 0 and print_detail:
                progbar_val.update(iter + 1, [('batch_cost', batch_cost),
                                              ('reader cost', reader_cost)])
            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

    class_iou, miou = metrics.mean_iou(intersect_area_all, pred_area_all,
                                       label_area_all)
    class_acc, acc = metrics.precision(intersect_area_all, pred_area_all)
    _, class_precision, class_rec = metrics.class_measurement(intersect_area_all, pred_area_all, label_area_all)
    kappa, po = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)
    # class_rec,mrec = metrics.recall(intersect_area_all,label_area_all)
    F1 = 2 * class_precision * class_rec / (class_precision + class_rec)
    meanF1 = np.mean(F1)
    if print_detail:
        logger.info("Client1: Using the Segformer model to process opt data")
        infor = "[EVAL] #Images: {} mIoU: {:.4f} Acc: {:.4f} Pre:{:.4f} Rec:{:.4f} F1:{:.4f} Kappa: {:.4f},poacc:{:.4f}".format(
            len(eval_dataset), miou, acc, np.mean(class_precision), np.mean(class_rec), meanF1, kappa, po)
        logger.info(infor)
        logger.info("[EVAL] Class IoU: \n" + str(np.round(class_iou, 4)))
        # logger.info("[EVAL] Class Acc: \n" + str(np.round(class_acc, 4)))
        logger.info("[EVAL] Class Pre: \n" + str(np.round(class_precision, 4)))
        logger.info("[EVAL] Class Rec: \n" + str(np.round(class_rec, 4)))
        logger.info("[EVAL] Class F1: \n" + str(np.round(F1, 4)))
    return miou, acc, F1, class_iou, class_acc, kappa


def evaluate_client2(model,
                     eval_dataset,
                     server2,
                     aug_eval=False,
                     scales=1.0,
                     flip_horizontal=True,
                     flip_vertical=False,
                     is_slide=False,
                     stride=None,
                     crop_size=None,
                     num_workers=0,
                     print_detail=True):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A sementic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        aug_eval (bool, optional): Whether to use mulit-scales and flip augment for evaluation. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_eval` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_eval` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_eval` is True. Default: False.
        is_slide (bool, optional): Whether to evaluate by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.

    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    model.eval()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
        ):
            paddle.distributed.init_parallel_env()
    batch_sampler = paddle.io.DistributedBatchSampler(
        eval_dataset, batch_size=1, shuffle=False, drop_last=False)
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
    )

    total_iters = len(loader)
    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0

    if print_detail:
        logger.info(
            "Start evaluating (total_samples: {}, total_iters: {})...".format(
                len(eval_dataset), total_iters))
    # TODO(chenguowei): fix log print error with multi-gpus
    progbar_val = progbar.Progbar(
        target=total_iters, verbose=1 if nranks < 2 else 2)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    with paddle.no_grad():
        for iter, (im, opt, dem, sar, label) in enumerate(loader):
            reader_cost_averager.record(time.time() - batch_start)
            label = label.astype('int64')
            dem_normalize = (dem - paddle.min(dem)) / (paddle.max(dem) - paddle.min(dem))
            sar_normalize = (sar - paddle.min(sar)) / (paddle.max(sar) - paddle.min(sar))
            ori_shape = label.shape[-2:]
            if aug_eval:
                pred = infer_server.aug_inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=eval_dataset.transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                # recv_file(server1, "rev_xy_files1_test.tensor")
                recv_file(server2, "rev_xy_files2_test.tensor")
                # recv_file(server3, "rev_xy_files3_test.tensor")
                # opt = paddle.load("rev_xy_files1_test.tensor")
                dem = paddle.load("rev_xy_files2_test.tensor")
                # sar = paddle.load("rev_xy_files3_test.tensor")
                # print(fusion.shape)

                pred = infer_server.inference_client2(
                    model,
                    dem,
                    ori_shape=ori_shape,
                    transforms=eval_dataset.transforms.transforms,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)

            intersect_area, pred_area, label_area = metrics.calculate_area(
                pred,
                label,
                eval_dataset.num_classes,
                ignore_index=eval_dataset.ignore_index)

            # Gather from all ranks
            if nranks > 1:
                intersect_area_list = []
                pred_area_list = []
                label_area_list = []
                paddle.distributed.all_gather(intersect_area_list,
                                              intersect_area)
                paddle.distributed.all_gather(pred_area_list, pred_area)
                paddle.distributed.all_gather(label_area_list, label_area)

                # Some image has been evaluated and should be eliminated in last iter
                if (iter + 1) * nranks > len(eval_dataset):
                    valid = len(eval_dataset) - iter * nranks
                    intersect_area_list = intersect_area_list[:valid]
                    pred_area_list = pred_area_list[:valid]
                    label_area_list = label_area_list[:valid]

                for i in range(len(intersect_area_list)):
                    intersect_area_all = intersect_area_all + intersect_area_list[
                        i]
                    pred_area_all = pred_area_all + pred_area_list[i]
                    label_area_all = label_area_all + label_area_list[i]
            else:
                intersect_area_all = intersect_area_all + intersect_area
                pred_area_all = pred_area_all + pred_area
                label_area_all = label_area_all + label_area
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            if local_rank == 0 and print_detail:
                progbar_val.update(iter + 1, [('batch_cost', batch_cost),
                                              ('reader cost', reader_cost)])
            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

    class_iou, miou = metrics.mean_iou(intersect_area_all, pred_area_all,
                                       label_area_all)
    class_acc, acc = metrics.precision(intersect_area_all, pred_area_all)
    _, class_precision, class_rec = metrics.class_measurement(intersect_area_all, pred_area_all, label_area_all)
    kappa, po = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)
    # class_rec,mrec = metrics.recall(intersect_area_all,label_area_all)
    F1 = 2 * class_precision * class_rec / (class_precision + class_rec)
    meanF1 = np.mean(F1)
    if print_detail:
        logger.info("Client2: Using the HRNet model to process DEM data")
        infor = "[EVAL] #Images: {} mIoU: {:.4f} Acc: {:.4f} Pre:{:.4f} Rec:{:.4f} F1:{:.4f} Kappa: {:.4f},poacc:{:.4f}".format(
            len(eval_dataset), miou, acc, np.mean(class_precision), np.mean(class_rec), meanF1, kappa, po)
        logger.info(infor)
        logger.info("[EVAL] Class IoU: \n" + str(np.round(class_iou, 4)))
        # logger.info("[EVAL] Class Acc: \n" + str(np.round(class_acc, 4)))
        logger.info("[EVAL] Class Pre: \n" + str(np.round(class_precision, 4)))
        logger.info("[EVAL] Class Rec: \n" + str(np.round(class_rec, 4)))
        logger.info("[EVAL] Class F1: \n" + str(np.round(F1, 4)))
    return miou, acc, F1, class_iou, class_acc, kappa


def evaluate_client3(model,
                     eval_dataset,
                     server3,
                     aug_eval=False,
                     scales=1.0,
                     flip_horizontal=True,
                     flip_vertical=False,
                     is_slide=False,
                     stride=None,
                     crop_size=None,
                     num_workers=0,
                     print_detail=True):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A sementic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        aug_eval (bool, optional): Whether to use mulit-scales and flip augment for evaluation. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_eval` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_eval` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_eval` is True. Default: False.
        is_slide (bool, optional): Whether to evaluate by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.

    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    model.eval()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
        ):
            paddle.distributed.init_parallel_env()
    batch_sampler = paddle.io.DistributedBatchSampler(
        eval_dataset, batch_size=1, shuffle=False, drop_last=False)
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
    )

    total_iters = len(loader)
    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0

    if print_detail:
        logger.info(
            "Start evaluating (total_samples: {}, total_iters: {})...".format(
                len(eval_dataset), total_iters))
    # TODO(chenguowei): fix log print error with multi-gpus
    progbar_val = progbar.Progbar(
        target=total_iters, verbose=1 if nranks < 2 else 2)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    with paddle.no_grad():
        for iter, (im, opt, dem, sar, label) in enumerate(loader):
            reader_cost_averager.record(time.time() - batch_start)
            label = label.astype('int64')
            dem_normalize = (dem - paddle.min(dem)) / (paddle.max(dem) - paddle.min(dem))
            sar_normalize = (sar - paddle.min(sar)) / (paddle.max(sar) - paddle.min(sar))
            ori_shape = label.shape[-2:]
            if aug_eval:
                pred = infer_server.aug_inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=eval_dataset.transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                # recv_file(server1, "rev_xy_files1_test.tensor")
                # recv_file(server2, "rev_xy_files2_test.tensor")
                recv_file(server3, "rev_xy_files3_test.tensor")
                # opt = paddle.load("rev_xy_files1_test.tensor")
                # dem = paddle.load("rev_xy_files2_test.tensor")
                sar = paddle.load("rev_xy_files3_test.tensor")
                # print(fusion.shape)

                pred = infer_server.inference_client3(
                    model,
                    sar,
                    ori_shape=ori_shape,
                    transforms=eval_dataset.transforms.transforms,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)

            intersect_area, pred_area, label_area = metrics.calculate_area(
                pred,
                label,
                eval_dataset.num_classes,
                ignore_index=eval_dataset.ignore_index)

            # Gather from all ranks
            if nranks > 1:
                intersect_area_list = []
                pred_area_list = []
                label_area_list = []
                paddle.distributed.all_gather(intersect_area_list,
                                              intersect_area)
                paddle.distributed.all_gather(pred_area_list, pred_area)
                paddle.distributed.all_gather(label_area_list, label_area)

                # Some image has been evaluated and should be eliminated in last iter
                if (iter + 1) * nranks > len(eval_dataset):
                    valid = len(eval_dataset) - iter * nranks
                    intersect_area_list = intersect_area_list[:valid]
                    pred_area_list = pred_area_list[:valid]
                    label_area_list = label_area_list[:valid]

                for i in range(len(intersect_area_list)):
                    intersect_area_all = intersect_area_all + intersect_area_list[
                        i]
                    pred_area_all = pred_area_all + pred_area_list[i]
                    label_area_all = label_area_all + label_area_list[i]
            else:
                intersect_area_all = intersect_area_all + intersect_area
                pred_area_all = pred_area_all + pred_area
                label_area_all = label_area_all + label_area
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            if local_rank == 0 and print_detail:
                progbar_val.update(iter + 1, [('batch_cost', batch_cost),
                                              ('reader cost', reader_cost)])
            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

    class_iou, miou = metrics.mean_iou(intersect_area_all, pred_area_all,
                                       label_area_all)
    class_acc, acc = metrics.precision(intersect_area_all, pred_area_all)
    _, class_precision, class_rec = metrics.class_measurement(intersect_area_all, pred_area_all, label_area_all)
    kappa, po = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)
    # class_rec,mrec = metrics.recall(intersect_area_all,label_area_all)
    F1 = 2 * class_precision * class_rec / (class_precision + class_rec)
    meanF1 = np.mean(F1)
    if print_detail:
        logger.info("Client3: Using the ResNet model to process sar data")
        infor = "[EVAL] #Images: {} mIoU: {:.4f} Acc: {:.4f} Pre:{:.4f} Rec:{:.4f} F1:{:.4f} Kappa: {:.4f},poacc:{:.4f}".format(
            len(eval_dataset), miou, acc, np.mean(class_precision), np.mean(class_rec), meanF1, kappa, po)
        logger.info(infor)
        logger.info("[EVAL] Class IoU: \n" + str(np.round(class_iou, 4)))
        # logger.info("[EVAL] Class Acc: \n" + str(np.round(class_acc, 4)))
        logger.info("[EVAL] Class Pre: \n" + str(np.round(class_precision, 4)))
        logger.info("[EVAL] Class Rec: \n" + str(np.round(class_rec, 4)))
        logger.info("[EVAL] Class F1: \n" + str(np.round(F1, 4)))
    return miou, acc, F1, class_iou, class_acc, kappa
