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
import math

import cv2
import numpy as np
import paddle

from paddleseg import utils
from paddleseg.core import infer_server, infer_client1, infer_client2, infer_client3
from paddleseg.utils import logger, progbar, visualize


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


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def predict(model,
            model_path,
            transforms,
            image_list,
            image_dir=None,
            save_dir='output',
            aug_pred=False,
            scales=1.0,
            flip_horizontal=True,
            flip_vertical=False,
            is_slide=False,
            stride=None,
            crop_size=None,
            custom_color=None):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_pred (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()
    # print(image_list)
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
    else:
        img_lists = [image_list]

    added_saved_dir = os.path.join(save_dir, 'added_prediction')
    pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')
    # print(image_list)
    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    color_map = visualize.get_color_map_list(256, custom_color=custom_color)
    with paddle.no_grad():
        print(img_lists)
        for i, (hill, opt, dem, sar) in enumerate(img_lists[local_rank]):
            im = cv2.imread(hill)
            ori_shape = im.shape[:2]
            im, _ = transforms(im)
            im = im[np.newaxis, ...]
            im = paddle.to_tensor(im)

            remote_img = cv2.imread(opt)
            remote_img, _ = transforms(remote_img)
            remote_img = remote_img[np.newaxis, ...]
            remote_img = paddle.to_tensor(remote_img)

            dem = cv2.imread(dem, cv2.IMREAD_ANYDEPTH or cv2.IMREAD_LOAD_GDAL)
            dem = cv2.resize(dem, (1024, 1024))
            dem = (dem - np.min(dem)) / (np.max(dem) - np.min(dem))
            dem = dem[np.newaxis, np.newaxis, :, :]
            dem = paddle.to_tensor(dem)

            sar = cv2.imread(sar, cv2.IMREAD_ANYDEPTH or cv2.IMREAD_LOAD_GDAL)
            sar = cv2.resize(sar, (1024, 1024))
            sar = (sar - np.min(sar)) / (np.max(sar) - np.min(sar))
            sar = sar[np.newaxis, np.newaxis, :, :]
            sar = paddle.to_tensor(sar)

            if aug_pred:
                pred = infer.aug_inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred = infer.inference(
                    model,
                    im,
                    remote_img,
                    dem,
                    sar,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            pred = paddle.squeeze(pred)
            pred = pred.numpy().astype('uint8')

            # get the saved name
            if image_dir is not None:
                im_file = hill.replace(image_dir, '')
            else:
                im_file = os.path.basename(hill)
            if im_file[0] == '/' or im_file[0] == '\\':
                im_file = im_file[1:]

            # save added image
            # added_image = utils.visualize.visualize(
            #     im_path, pred, color_map, weight=0.6)
            # added_image_path = os.path.join(added_saved_dir, im_file)
            # mkdir(added_image_path)
            # cv2.imwrite(added_image_path, added_image)

            # save pseudo color prediction
            pred_mask = utils.visualize.get_pseudo_color_map(pred, color_map)
            pred_saved_path = os.path.join(
                pred_saved_dir,
                os.path.splitext(im_file)[0] + ".png")
            mkdir(pred_saved_path)
            pred_mask.save(pred_saved_path)

            # pred_im = utils.visualize(im_path, pred, weight=0.0)
            # pred_saved_path = os.path.join(pred_saved_dir, im_file)
            # mkdir(pred_saved_path)
            # cv2.imwrite(pred_saved_path, pred_im)

            progbar_pred.update(i + 1)


def predict_server(model,
                   server1,
                   server2,
                   server3,
                   model_path,
                   transforms,
                   image_list,
                   image_dir=None,
                   save_dir='output',
                   aug_pred=False,
                   scales=1.0,
                   flip_horizontal=True,
                   flip_vertical=False,
                   is_slide=False,
                   stride=None,
                   crop_size=None,
                   custom_color=None):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_pred (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()
    # print(image_list)
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
    else:
        img_lists = [image_list]

    added_saved_dir = os.path.join(save_dir, 'added_prediction')
    pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')
    # print(image_list)
    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    color_map = visualize.get_color_map_list(256, custom_color=custom_color)
    with paddle.no_grad():
        # print(img_lists)
        for i, (hill, opt, dem, sar) in enumerate(img_lists[local_rank]):
            im = cv2.imread(hill)
            ori_shape = im.shape[:2]
            im, _ = transforms(im)
            im = im[np.newaxis, ...]
            im = paddle.to_tensor(im)

            remote_img = cv2.imread(opt)
            remote_img, _ = transforms(remote_img)
            remote_img = remote_img[np.newaxis, ...]
            remote_img = paddle.to_tensor(remote_img)

            dem = cv2.imread(dem, cv2.IMREAD_ANYDEPTH or cv2.IMREAD_LOAD_GDAL)
            dem = cv2.resize(dem, (1024, 1024))
            dem = (dem - np.min(dem)) / (np.max(dem) - np.min(dem))
            dem = dem[np.newaxis, np.newaxis, :, :]
            dem = paddle.to_tensor(dem)

            sar = cv2.imread(sar, cv2.IMREAD_ANYDEPTH or cv2.IMREAD_LOAD_GDAL)
            sar = cv2.resize(sar, (1024, 1024))
            sar = (sar - np.min(sar)) / (np.max(sar) - np.min(sar))
            sar = sar[np.newaxis, np.newaxis, :, :]
            sar = paddle.to_tensor(sar)

            if aug_pred:
                pred = infer.aug_inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                recv_file(server1, "rev_xy_files1_predict.tensor")
                send_cmd(server1, '1')
                recv_file(server2, "rev_xy_files2_predict.tensor")
                send_cmd(server2, '1')
                recv_file(server3, "rev_xy_files3_predict.tensor")
                send_cmd(server3, '1')
                opt = paddle.load("rev_xy_files1_predict.tensor")
                dem = paddle.load("rev_xy_files2_predict.tensor")
                hillshade = paddle.load("rev_xy_files3_predict.tensor")
                pred = infer_server.inference(
                    model,
                    opt,
                    dem,
                    hillshade,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            pred = paddle.squeeze(pred)
            pred = pred.numpy().astype('uint8')

            # get the saved name
            if image_dir is not None:
                im_file = hill.replace(image_dir, '')
            else:
                im_file = os.path.basename(hill)
            if im_file[0] == '/' or im_file[0] == '\\':
                im_file = im_file[1:]

            # save added image
            # added_image = utils.visualize.visualize(
            #     im_path, pred, color_map, weight=0.6)
            # added_image_path = os.path.join(added_saved_dir, im_file)
            # mkdir(added_image_path)
            # cv2.imwrite(added_image_path, added_image)

            # save pseudo color prediction
            pred_mask = utils.visualize.get_pseudo_color_map(pred, color_map)
            pred_saved_path = os.path.join(
                pred_saved_dir,
                os.path.splitext(im_file)[0] + ".png")
            mkdir(pred_saved_path)
            pred_mask.save(pred_saved_path)

            # pred_im = utils.visualize(im_path, pred, weight=0.0)
            # pred_saved_path = os.path.join(pred_saved_dir, im_file)
            # mkdir(pred_saved_path)
            # cv2.imwrite(pred_saved_path, pred_im)

            progbar_pred.update(i + 1)


def predict_client1(model,
                    client,
                    model_path,
                    transforms,
                    image_list,
                    image_dir=None,
                    save_dir='output',
                    aug_pred=False,
                    scales=1.0,
                    flip_horizontal=True,
                    flip_vertical=False,
                    is_slide=False,
                    stride=None,
                    crop_size=None,
                    custom_color=None):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_pred (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()
    # print(image_list)
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
    else:
        img_lists = [image_list]

    # added_saved_dir = os.path.join(save_dir, 'added_prediction')
    # pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')
    # print(image_list)
    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    color_map = visualize.get_color_map_list(256, custom_color=custom_color)
    with paddle.no_grad():
        # print(img_lists)
        for i, (hill, opt, dem, sar) in enumerate(img_lists[local_rank]):
            im = cv2.imread(hill)
            ori_shape = im.shape[:2]
            im, _ = transforms(im)
            im = im[np.newaxis, ...]
            im = paddle.to_tensor(im)

            remote_img = cv2.imread(opt)
            remote_img, _ = transforms(remote_img)
            remote_img = remote_img[np.newaxis, ...]
            remote_img = paddle.to_tensor(remote_img)

            dem = cv2.imread(dem, cv2.IMREAD_ANYDEPTH or cv2.IMREAD_LOAD_GDAL)
            dem = cv2.resize(dem, (1024, 1024))
            dem = (dem - np.min(dem)) / (np.max(dem) - np.min(dem))
            dem = dem[np.newaxis, np.newaxis, :, :]
            dem = paddle.to_tensor(dem)

            sar = cv2.imread(sar, cv2.IMREAD_ANYDEPTH or cv2.IMREAD_LOAD_GDAL)
            sar = cv2.resize(sar, (1024, 1024))
            sar = (sar - np.min(sar)) / (np.max(sar) - np.min(sar))
            sar = sar[np.newaxis, np.newaxis, :, :]
            sar = paddle.to_tensor(sar)

            if aug_pred:
                pred = infer.aug_inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred = infer_client1.inference(
                    model,
                    remote_img,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
                paddle.save(pred, "xy_files1_predict.tensor")
                # print(pred.shape)
                send_file(client, "xy_files1_predict.tensor")
                recv_cmd(client)
            # pred = paddle.squeeze(pred)
            # pred = pred.numpy().astype('uint8')
            #
            # # get the saved name
            # if image_dir is not None:
            #     im_file = hill.replace(image_dir, '')
            # else:
            #     im_file = os.path.basename(hill)
            # if im_file[0] == '/' or im_file[0] == '\\':
            #     im_file = im_file[1:]
            #
            # # save added image
            # # added_image = utils.visualize.visualize(
            # #     im_path, pred, color_map, weight=0.6)
            # # added_image_path = os.path.join(added_saved_dir, im_file)
            # # mkdir(added_image_path)
            # # cv2.imwrite(added_image_path, added_image)
            #
            # # save pseudo color prediction
            # pred_mask = utils.visualize.get_pseudo_color_map(pred, color_map)
            # pred_saved_path = os.path.join(
            #     pred_saved_dir,
            #     os.path.splitext(im_file)[0] + ".png")
            # mkdir(pred_saved_path)
            # pred_mask.save(pred_saved_path)

            # pred_im = utils.visualize(im_path, pred, weight=0.0)
            # pred_saved_path = os.path.join(pred_saved_dir, im_file)
            # mkdir(pred_saved_path)
            # cv2.imwrite(pred_saved_path, pred_im)

            progbar_pred.update(i + 1)


def predict_client2(model,
                    client,
                    model_path,
                    transforms,
                    image_list,
                    image_dir=None,
                    save_dir='output',
                    aug_pred=False,
                    scales=1.0,
                    flip_horizontal=True,
                    flip_vertical=False,
                    is_slide=False,
                    stride=None,
                    crop_size=None,
                    custom_color=None):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_pred (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()
    # print(image_list)
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
    else:
        img_lists = [image_list]

    # added_saved_dir = os.path.join(save_dir, 'added_prediction')
    # pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')
    # print(image_list)
    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    color_map = visualize.get_color_map_list(256, custom_color=custom_color)
    with paddle.no_grad():
        # print(img_lists)
        for i, (hill, opt, dem, sar) in enumerate(img_lists[local_rank]):
            im = cv2.imread(hill)
            ori_shape = im.shape[:2]
            im, _ = transforms(im)
            im = im[np.newaxis, ...]
            im = paddle.to_tensor(im)

            remote_img = cv2.imread(opt)
            remote_img, _ = transforms(remote_img)
            remote_img = remote_img[np.newaxis, ...]
            remote_img = paddle.to_tensor(remote_img)

            dem = cv2.imread(dem, cv2.IMREAD_ANYDEPTH or cv2.IMREAD_LOAD_GDAL)
            dem = cv2.resize(dem, (1024, 1024))
            dem = (dem - np.min(dem)) / (np.max(dem) - np.min(dem))
            dem = dem[np.newaxis, np.newaxis, :, :]
            dem = paddle.to_tensor(dem)

            sar = cv2.imread(sar, cv2.IMREAD_ANYDEPTH or cv2.IMREAD_LOAD_GDAL)
            sar = cv2.resize(sar, (1024, 1024))
            sar = (sar - np.min(sar)) / (np.max(sar) - np.min(sar))
            sar = sar[np.newaxis, np.newaxis, :, :]
            sar = paddle.to_tensor(sar)

            if aug_pred:
                pred = infer.aug_inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred = infer_client2.inference(
                    model,
                    dem,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
                paddle.save(pred, "xy_files2_predict.tensor")
                # print(pred.shape)
                send_file(client, "xy_files2_predict.tensor")
                recv_cmd(client)
            # pred = paddle.squeeze(pred)
            # pred = pred.numpy().astype('uint8')
            #
            # # get the saved name
            # if image_dir is not None:
            #     im_file = hill.replace(image_dir, '')
            # else:
            #     im_file = os.path.basename(hill)
            # if im_file[0] == '/' or im_file[0] == '\\':
            #     im_file = im_file[1:]
            #
            # # save added image
            # # added_image = utils.visualize.visualize(
            # #     im_path, pred, color_map, weight=0.6)
            # # added_image_path = os.path.join(added_saved_dir, im_file)
            # # mkdir(added_image_path)
            # # cv2.imwrite(added_image_path, added_image)
            #
            # # save pseudo color prediction
            # pred_mask = utils.visualize.get_pseudo_color_map(pred, color_map)
            # pred_saved_path = os.path.join(
            #     pred_saved_dir,
            #     os.path.splitext(im_file)[0] + ".png")
            # mkdir(pred_saved_path)
            # pred_mask.save(pred_saved_path)
            #
            # # pred_im = utils.visualize(im_path, pred, weight=0.0)
            # # pred_saved_path = os.path.join(pred_saved_dir, im_file)
            # # mkdir(pred_saved_path)
            # # cv2.imwrite(pred_saved_path, pred_im)

            progbar_pred.update(i + 1)


def predict_client3(model,
                    client,
                    model_path,
                    transforms,
                    image_list,
                    image_dir=None,
                    save_dir='output',
                    aug_pred=False,
                    scales=1.0,
                    flip_horizontal=True,
                    flip_vertical=False,
                    is_slide=False,
                    stride=None,
                    crop_size=None,
                    custom_color=None):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_pred (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()
    # print(image_list)
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
    else:
        img_lists = [image_list]

    # added_saved_dir = os.path.join(save_dir, 'added_prediction')
    # pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')
    # print(image_list)
    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    color_map = visualize.get_color_map_list(256, custom_color=custom_color)
    with paddle.no_grad():
        # print(img_lists)
        for i, (hill, opt, dem, sar) in enumerate(img_lists[local_rank]):
            im = cv2.imread(hill)
            ori_shape = im.shape[:2]
            im, _ = transforms(im)
            im = im[np.newaxis, ...]
            im = paddle.to_tensor(im)

            remote_img = cv2.imread(opt)
            remote_img, _ = transforms(remote_img)
            remote_img = remote_img[np.newaxis, ...]
            remote_img = paddle.to_tensor(remote_img)

            dem = cv2.imread(dem, cv2.IMREAD_ANYDEPTH or cv2.IMREAD_LOAD_GDAL)
            dem = cv2.resize(dem, (1024, 1024))
            dem = (dem - np.min(dem)) / (np.max(dem) - np.min(dem))
            dem = dem[np.newaxis, np.newaxis, :, :]
            dem = paddle.to_tensor(dem)

            sar = cv2.imread(sar, cv2.IMREAD_ANYDEPTH or cv2.IMREAD_LOAD_GDAL)
            sar = cv2.resize(sar, (1024, 1024))
            sar = (sar - np.min(sar)) / (np.max(sar) - np.min(sar))
            sar = sar[np.newaxis, np.newaxis, :, :]
            sar = paddle.to_tensor(sar)

            if aug_pred:
                pred = infer.aug_inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred = infer_client3.inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
                paddle.save(pred, "xy_files3_predict.tensor")
                # print(pred.shape)
                send_file(client, "xy_files3_predict.tensor")
                recv_cmd(client)
            # pred = paddle.squeeze(pred)
            # pred = pred.numpy().astype('uint8')
            #
            # # get the saved name
            # if image_dir is not None:
            #     im_file = hill.replace(image_dir, '')
            # else:
            #     im_file = os.path.basename(hill)
            # if im_file[0] == '/' or im_file[0] == '\\':
            #     im_file = im_file[1:]
            #
            # # save added image
            # # added_image = utils.visualize.visualize(
            # #     im_path, pred, color_map, weight=0.6)
            # # added_image_path = os.path.join(added_saved_dir, im_file)
            # # mkdir(added_image_path)
            # # cv2.imwrite(added_image_path, added_image)
            #
            # # save pseudo color prediction
            # pred_mask = utils.visualize.get_pseudo_color_map(pred, color_map)
            # pred_saved_path = os.path.join(
            #     pred_saved_dir,
            #     os.path.splitext(im_file)[0] + ".png")
            # mkdir(pred_saved_path)
            # pred_mask.save(pred_saved_path)
            #
            # # pred_im = utils.visualize(im_path, pred, weight=0.0)
            # # pred_saved_path = os.path.join(pred_saved_dir, im_file)
            # # mkdir(pred_saved_path)
            # # cv2.imwrite(pred_saved_path, pred_im)

            progbar_pred.update(i + 1)
