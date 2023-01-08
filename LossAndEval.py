import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, jaccard_score, f1_score, accuracy_score

# PyTorch
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        target = (target > 0.5).float()

        return self.bce(input, target)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target, weight=1):

        input = F.sigmoid(input)

        return 1-dice_coef(input, target, weight=weight)


class DiceLoss2(nn.Module):
    def __init__(self):
        super(DiceLoss2, self).__init__()

    def forward(self, input, target, weight=1):

        input = F.sigmoid(input)

        return 1-dice_coef(input, target, weight=weight, isPow=True)

class Dice_BCELoss(nn.Module):
    def __init__(self):
        super(Dice_BCELoss, self).__init__()

    def forward(self, input, target, weight=1):
        input_ = F.sigmoid(input)

        dice_coef_loss = dice_coef(input_, target, weight=weight)

        bce = nn.BCEWithLogitsLoss()
        bce_loss = bce(input, target)

        loss = bce_loss - dice_coef_loss
        return loss

def dice_coef(input, target, weight=1, isPow=False):
    smooth = 1e-5
    target = (target > 0.5).float()
    input_flat = input.view(-1)
    target_flat = target.view(-1)
    intersection = (input_flat * target_flat)

    if isPow:
        return (2 * weight * intersection.sum() + smooth) / (input_flat.pow(2).sum() + target_flat.pow(2).sum() + smooth)
    else:
        return (2 * weight * intersection.sum() + smooth) / (input_flat.sum() + target_flat.sum() + smooth)


def iou_score(output, target):
    smooth = 1e-5

    output = output.view(-1)
    target = target.view(-1)

    output_ = output > 0.5
    target_ = target > 0.5

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def precision(output, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        f1: float
    """
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output = output > 0.5
    target = target > 0.5

    output.tolist(), target.tolist()
    output, target = np.array(output).flatten(), np.array(target).flatten()

    # output_flat = output.view(-1)
    # target_flat = target.view(-1)

    # print(111)
    precision_ = precision_score(y_true=target, y_pred=output)
    # precision_ = precision_score(y_true=target_flat, y_pred=output_flat)
    #
    # print(precision_)
    return precision_

def acc(output, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        f1: float
    """
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output = output > 0.5
    target = target > 0.5

    output.tolist(), target.tolist()
    output, target = np.array(output).flatten(), np.array(target).flatten()

    acc_ = accuracy_score(y_true=target, y_pred=output)
    return acc_

def recall(output, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        f1: float
    """
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output = output > 0.5
    target = target > 0.5

    output.tolist(), target.tolist()
    output, target = np.array(output).flatten(), np.array(target).flatten()

    recall_ = recall_score(y_true=target, y_pred=output)

    return recall_

def jaccard(output, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        f1: float
    """
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output = output > 0.5
    target = target > 0.5

    output.tolist(), target.tolist()
    output, target = np.array(output).flatten(), np.array(target).flatten()

    jaccard_ = jaccard_score(y_true=target, y_pred=output)

    return jaccard_


def sensitivity_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()  # 0,1之间
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / \
           (target.sum() + smooth)


def get_sens(input,target):
    smooth = 1e-5
    input_flat = input.view(-1)
    target_flat = target.view(-1)
    intersection = (input_flat * target_flat)

    result = (intersection.sum() + smooth)/(target_flat.sum() + smooth)

    return result

def get_ppv(input,target):
    smooth = 1e-5
    input_flat = input.view(-1)
    target_flat = target.view(-1)
    intersection = (input_flat * target_flat)

    result = (intersection.sum() + smooth) / (input_flat.sum() + smooth)

    return result

def ppv_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / \
           (output.sum() + smooth)

def f1score(output, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        f1: float
    """
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output.tolist(), target.tolist()
    output, target = np.array(output).flatten(), np.array(target).flatten()
    f1 = f1_score(y_true=target, y_pred=output)
    return  f1

# def computeQualityMeasures(output, target):
#     import SimpleITK as sitk
#     # output = output.unsqueeze(dim=1)
#     # target = target.unsqueeze(dim=1)
#
#     # output = (output > 0.5).float()
#     # target = (target > 0.5).float()
#
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#
#     # channel = output.shape[0]
#
#     # for i in range(channel):
#     # print(output[i].shape)
#     quality = dict()
#     output_ = sitk.GetImageFromArray(output, isVector=False)
#     target_ = sitk.GetImageFromArray(target, isVector=False)
#
#     hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
#     hausdorffcomputer.Execute(target_ > 0.5, output_ > 0.5)
#     quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
#     quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()
#     print(quality["Hausdorff"])

    # return quality["Hausdorff"]


