import torch
import torch.nn as nn

from mmdet.ops import convex_sort
from mmdet.core import bbox2type, get_bbox_areas
from mmdet.models.builder import LOSSES
from ..utilsv2 import weighted_lossv2

@weighted_lossv2
def box_points_loss(bpts_pred, bbox_target, eps=1e-6):
    # Calculate target areas
    target_area = poly_area(bbox_target)

    # Calculate box point areas
    area_targs = [rectangle_area(bbox_target, bpts_pred[:, i:i+2]) for i in range(0, 16, 2)]

    # Calculate kernel values
    K_targs = [kernel(area_targ, target_area) for area_targ in area_targs]

    # Calculate intersection
    intersection = sum(K_targs)

    # Calculate iou and loss
    ious = torch.abs(intersection / 8)
    loss = 1 - ious

    return loss

def kernel(pred, target, k=4):
    assert pred.shape == target.shape
    x = torch.abs((pred - target) / target)
    kf = 2 / (1 + torch.exp(k * x))
    return kf

def rectangle_area(bbox, bpts):
    area = sum([triangle_area(bbox[:, i:i+2], bbox[:, (i+2) % 8:i+4], bpts) for i in range(0, 8, 2)])
    return area

def poly_area(poly):
    area = triangle_area(poly[:, 0:2], poly[:, 2:4], poly[:, 4:6]) + triangle_area(poly[:, 2:4], poly[:, 6:8], poly[:, 4:6])
    return area

def triangle_area(point_one, point_two, bpts_point):
    part_1 = point_one[:, 0]*bpts_point[:, 1] + bpts_point[:, 0]*point_two[:, 1] + point_two[:, 0]*point_one[:, 1]
    part_2 = point_one[:, 1]*bpts_point[:, 0] + bpts_point[:, 1]*point_two[:, 0] + point_two[:, 1]*point_one[:, 0]

    area = torch.abs((part_1 - part_2)/2)
    return area

@LOSSES.register_module()
class BoxPointsLoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(BoxPointsLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, bpts, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        
        if (weight is not None) and (not torch.any(weight > 0)) and (reduction != 'none'):
            return (bpts * weight).sum()  # 0
        
        if weight is not None and weight.dim() > 1:
            assert weight.shape[0] == bpts.shape[0]
            weight = weight.mean(-1)
        
        loss = self.loss_weight * box_points_loss(bpts, target, weight, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss
