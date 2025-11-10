# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from lib.utils.utils import _transpose_and_gather_feat, _sigmoid
import torch.nn.functional as F
import math

class LBHingev2(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, threshold=None, return_per_sequence=False):
        super().__init__()
        self.threshold = threshold if threshold is not None else -100
        self.return_per_sequence = return_per_sequence

    def forward(self, pred, gt):
        assert pred.dim() == 4 and gt.dim() == 4

        neg_inds = gt.lt(0.05)
        pos_inds = gt.gt(0.05)

        pred_relu_mask = pred.gt(0).float()
        pos_mask = pos_inds.float()
        neg_mask = 1.0 - pos_mask

        pred_new = neg_mask * pred_relu_mask * pred + pos_mask * pred

        # Mask invalid samples
        loss = F.mse_loss(pred_new, gt * pos_mask)

        # loss = loss / (all_pos_mask.sum() + 1e-4)

        return loss

def _slow_neg_loss(pred, gt):
    '''focal loss from CornerNet'''
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos + 1e-4
    return loss


def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    num_pos = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -= all_loss
    return loss


def _slow_reg_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr[mask]
    gt_regr = gt_regr[mask]

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
      Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _get_target_bbox(target_wh, target_reg, ind, mask, width, down_ratio):
    """
    根据标注信息计算真实框的坐标。
    target_wh, target_reg 是特征图尺度上的目标信息。
    """
    # target_wh, target_reg 已经是 [batch, num_points, 2] 的形式
    # 如果不是，需要根据实际数据格式修改
    ct_x = (ind % width).float()
    ct_y = torch.div(ind, width, rounding_mode='trunc').float()

    target_center_x = ct_x + target_reg[..., 0]
    target_center_y = ct_y + target_reg[..., 1]

    target_w = target_wh[..., 0]
    target_h = target_wh[..., 1]

    x1 = target_center_x - target_w / 2
    y1 = target_center_y - target_h / 2
    x2 = target_center_x + target_w / 2
    y2 = target_center_y + target_h / 2

    # 回到原图坐标系
    x1 *= down_ratio
    y1 *= down_ratio
    x2 *= down_ratio
    y2 *= down_ratio

    target_bbox = torch.stack([x1, y1, x2, y2], dim=2) # [batch, num_points, 4]
    return target_bbox

def _get_bbox(pred_wh, pred_reg, ind, mask, width, down_ratio):
    """
    根据特征图上的预测结果计算预测框的坐标。
    假设特征图大小为 height x width，原图下采样为 down_ratio。
    pred_wh, pred_reg: 特征图尺度上的预测结果
    """
    # 提取特定点位置的预测值
    pred_wh = _transpose_and_gather_feat(pred_wh, ind)   # [batch, num_points, 2]
    pred_reg = _transpose_and_gather_feat(pred_reg, ind) # [batch, num_points, 2]

    # 计算格点坐标（特征图尺度）
    ct_x = (ind % width).float()
    ct_y = torch.div(ind, width, rounding_mode='trunc').float()

    # 中心点坐标 = 格点坐标 + 偏移量
    pred_center_x = ct_x + pred_reg[..., 0]
    pred_center_y = ct_y + pred_reg[..., 1]

    # 获取预测的宽高
    pred_w = pred_wh[..., 0]
    pred_h = pred_wh[..., 1]

    # 计算边界框坐标（特征图尺度）
    x1 = pred_center_x - pred_w / 2
    y1 = pred_center_y - pred_h / 2
    x2 = pred_center_x + pred_w / 2
    y2 = pred_center_y + pred_h / 2

    # 如果需要回到原图坐标系，请乘以 down_ratio
    x1 *= down_ratio
    y1 *= down_ratio
    x2 *= down_ratio
    y2 *= down_ratio

    pred_bbox = torch.stack([x1, y1, x2, y2], dim=2)  # [batch, num_points, 4]
    return pred_bbox

class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class RegLoss(nn.Module):
    '''Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss
class CIoULoss(nn.Module):
    def __init__(self):
        super(CIoULoss, self).__init__()

    def forward(self, pred_boxes, target_boxes, mask=None):
        """
        pred_boxes: [batch, num_points, 4], [x1, y1, x2, y2]
        target_boxes: [batch, num_points, 4], [x1, y1, x2, y2]
        """
        if mask is not None:
            # 确保 mask 的形状为 [batch, num_points, 1]
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)  # [batch, num_points, 1]
            mask = mask.float()

        inter_x1 = torch.max(pred_boxes[..., 0], target_boxes[..., 0])
        inter_y1 = torch.max(pred_boxes[..., 1], target_boxes[..., 1])
        inter_x2 = torch.min(pred_boxes[..., 2], target_boxes[..., 2])
        inter_y2 = torch.min(pred_boxes[..., 3], target_boxes[..., 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        target_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])

        union_area = pred_area + target_area - inter_area + 1e-6
        iou = inter_area / union_area

        # 中心点距离
        pred_center_x = (pred_boxes[..., 0] + pred_boxes[..., 2]) / 2
        pred_center_y = (pred_boxes[..., 1] + pred_boxes[..., 3]) / 2
        target_center_x = (target_boxes[..., 0] + target_boxes[..., 2]) / 2
        target_center_y = (target_boxes[..., 1] + target_boxes[..., 3]) / 2

        center_distance = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2

        # 最小闭包包围框对角线长度
        enc_x1 = torch.min(pred_boxes[..., 0], target_boxes[..., 0])
        enc_y1 = torch.min(pred_boxes[..., 1], target_boxes[..., 1])
        enc_x2 = torch.max(pred_boxes[..., 2], target_boxes[..., 2])
        enc_y2 = torch.max(pred_boxes[..., 3], target_boxes[..., 3])

        enc_diagonal = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-6

        # 长宽比一致性
        pred_w = pred_boxes[..., 2] - pred_boxes[..., 0]
        pred_h = pred_boxes[..., 3] - pred_boxes[..., 1]
        target_w = target_boxes[..., 2] - target_boxes[..., 0]
        target_h = target_boxes[..., 3] - target_boxes[..., 1]

        v = (4 / (math.pi ** 2)) * (torch.atan(target_w / (target_h + 1e-6)) - torch.atan(pred_w / (pred_h + 1e-6))) ** 2
        alpha = v / ((1 - iou) + v + 1e-6)

        ciou = iou - (center_distance / enc_diagonal) - alpha * v
        loss = 1 - ciou
        if mask is not None:
            # 仅对 mask 为1的位置计算损失
            loss = loss * mask.squeeze(-1)  # [batch, num_points]
            # 对有效位置求平均
            loss = loss.sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        return loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class NormRegL1Loss(nn.Module):
    def __init__(self):
        super(NormRegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        pred = pred / (target + 1e-4)
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        return loss


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _transpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')


# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')


def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res
