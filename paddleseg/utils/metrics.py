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

import numpy as np
import paddle
import paddle.nn.functional as F

def calculate_area(pred, label, num_classes, ignore_index=255):
    """
    Calculate intersect, prediction and label area

    Args:
        pred (Tensor): The prediction by model.
        label (Tensor): The ground truth of image.
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.

    Returns:
        Tensor: The intersection area of prediction and the ground on all class.
        Tensor: The prediction area on all class.
        Tensor: The ground truth area on all class
    """
    if len(pred.shape) == 4:
        pred = paddle.squeeze(pred, axis=1)
    if len(label.shape) == 4:
        label = paddle.squeeze(label, axis=1)
    if not pred.shape == label.shape:
        raise ValueError('Shape of `pred` and `label should be equal, '
                         'but there are {} and {}.'.format(
                             pred.shape, label.shape))
    pred_area = []
    label_area = []
    intersect_area = []
    mask = label != ignore_index

    for i in range(num_classes):
        pred_i = paddle.logical_and(pred == i, mask)
        label_i = label == i
        intersect_i = paddle.logical_and(pred_i, label_i)
        pred_area.append(paddle.sum(paddle.cast(pred_i, "int32")))
        label_area.append(paddle.sum(paddle.cast(label_i, "int32")))
        intersect_area.append(paddle.sum(paddle.cast(intersect_i, "int32")))

    pred_area = paddle.concat(pred_area)
    label_area = paddle.concat(label_area)
    intersect_area = paddle.concat(intersect_area)

    return intersect_area, pred_area, label_area
def mean_iou(intersect_area, pred_area, label_area):
    """
    Calculate iou.

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.

    Returns:
        np.ndarray: iou on all classes.
        float: mean iou of all classes.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    union = pred_area + label_area - intersect_area
    class_iou = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            iou = 0
        else:
            iou = intersect_area[i] / union[i]
        class_iou.append(iou)
    miou = np.mean(class_iou)
    return np.array(class_iou), miou


def precision(intersect_area, pred_area):
    """
    Calculate accuracy

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes..
        pred_area (Tensor): The prediction area on all classes.

    Returns:
        np.ndarray: accuracy on all classes.
        float: mean accuracy.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    class_acc = []
    for i in range(len(intersect_area)):
        if pred_area[i] == 0:
            acc = 0
        else:
            acc = intersect_area[i] / pred_area[i]
        class_acc.append(acc)
    macc = np.sum(intersect_area) / np.sum(pred_area)
    # macc = np.mean(class_acc)
    return np.array(class_acc), macc


def class_measurement(intersect_area, pred_area, label_area):
    """
    Calculate accuracy, class precision, and class recall.

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.

    Returns:
        float: The mean accuracy.
        np.ndarray: The precision of all classes.
        np.ndarray: The recall of all classes.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()

    mean_acc = np.sum(intersect_area) / np.sum(pred_area)
    class_precision = []
    class_recall = []
    for i in range(len(intersect_area)):
        precision = intersect_area[i] / max(pred_area[i], 1)  # Avoid division by zero
        recall = intersect_area[i] / max(label_area[i], 1)  # Avoid division by zero
        class_precision.append(precision)
        class_recall.append(recall)

    return mean_acc, np.array(class_precision), np.array(class_recall)


# def class_measurement(intersect_area, pred_area, label_area):
#     """
#     Calculate accuracy, calss precision and class recall.
#     Args:
#         intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
#         pred_area (Tensor): The prediction area on all classes.
#         label_area (Tensor): The ground truth area on all classes.
#     Returns:
#         float: The mean accuracy.
#         np.ndarray: The precision of all classes.
#         np.ndarray: The recall of all classes.
#     """
#     intersect_area = intersect_area.numpy()
#     pred_area = pred_area.numpy()
#     label_area = label_area.numpy()
#
#     mean_acc = np.sum(intersect_area) / np.sum(pred_area)
#     class_precision = []
#     class_recall = []
#     for i in range(len(intersect_area)):
#         precision = 0 if pred_area[i] == 0 else intersect_area[i] / pred_area[i]
#         recall = 0 if label_area[i] == 0 else intersect_area[i] / label_area[i]
#         class_precision.append(precision)
#         class_recall.append(recall)
#
#     return mean_acc, np.array(class_precision), np.array(class_recall)


# def recall(intersect_area,label_area):
#     intersect_area = intersect_area.numpy()
#     label_area = label_area.numpy()
#     class_rec = []
#     for i in range(len(intersect_area)):
#         rec = intersect_area[i] / label_area[i]
#         class_rec.append(rec)
#     mrec = np.mean(class_rec)
#     return np.array(class_rec), mrec

def kappa(intersect_area, pred_area, label_area):
    """
    Calculate kappa coefficient

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes..
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.

    Returns:
        float: kappa coefficient.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    total_area = np.sum(label_area)
    po = np.sum(intersect_area) / total_area
    pe = np.sum(pred_area * label_area) / (total_area * total_area)
    kappa = (po - pe) / (1 - pe)
    return kappa,po


