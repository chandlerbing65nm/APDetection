import torch
import torch.nn as nn

from mmdet.ops import convex_sort
from mmdet.core import bbox2type, get_bbox_areas
from mmdet.models.builder import LOSSES
from ..utilsv2 import weighted_lossv2

@weighted_lossv2
def box_points_loss(bpts_pred,
                    # bbox_pred,
                    bbox_target, 
                    eps=1e-6):
    
    # compute the area of the bbox target and bbox pred    
    target_area = poly_area(bbox_target)
    # pred_area = poly_area(bbox_pred)

    # compute the area of the box points wrt to bbox targets   
    area_targ1 = rectangle_area(bbox_target, bpts_pred[:, 0:2])
    area_targ2 = rectangle_area(bbox_target, bpts_pred[:, 2:4])
    area_targ3 = rectangle_area(bbox_target, bpts_pred[:, 4:6])
    area_targ4 = rectangle_area(bbox_target, bpts_pred[:, 6:8])
    area_targ5 = rectangle_area(bbox_target, bpts_pred[:, 8:10])
    area_targ6 = rectangle_area(bbox_target, bpts_pred[:, 10:12])
    area_targ7 = rectangle_area(bbox_target, bpts_pred[:, 12:14])
    area_targ8 = rectangle_area(bbox_target, bpts_pred[:, 14:16])

    # compute the area of the box points wrt to bbox pred	
    # area_pred1 = rectangle_area(bbox_pred, bpts_pred[:, 0:2])	
    # area_pred2 = rectangle_area(bbox_pred, bpts_pred[:, 2:4])	
    # area_pred3 = rectangle_area(bbox_pred, bpts_pred[:, 4:6])	
    # area_pred4 = rectangle_area(bbox_pred, bpts_pred[:, 6:8])	
    # area_pred5 = rectangle_area(bbox_pred, bpts_pred[:, 8:10])	
    # area_pred6 = rectangle_area(bbox_pred, bpts_pred[:, 10:12])	
    # area_pred7 = rectangle_area(bbox_pred, bpts_pred[:, 12:14])	
    # area_pred8 = rectangle_area(bbox_pred, bpts_pred[:, 14:16])

    # compare the bpts area and bbox target areas using sigmoid function
    K_targ1 = kernel(area_targ1, target_area)
    K_targ2 = kernel(area_targ2, target_area)
    K_targ3 = kernel(area_targ3, target_area)
    K_targ4 = kernel(area_targ4, target_area)
    K_targ5 = kernel(area_targ5, target_area)
    K_targ6 = kernel(area_targ6, target_area)
    K_targ7 = kernel(area_targ7, target_area)
    K_targ8 = kernel(area_targ8, target_area)

    # compare the bpts area and bbox pred areas using sigmoid function	
    # K_pred1 = kernel(area_pred1, pred_area)	
    # K_pred2 = kernel(area_pred2, pred_area)	
    # K_pred3 = kernel(area_pred3, pred_area)	
    # K_pred4 = kernel(area_pred4, pred_area)	
    # K_pred5 = kernel(area_pred5, pred_area)	
    # K_pred6 = kernel(area_pred6, pred_area)	
    # K_pred7 = kernel(area_pred7, pred_area)	
    # K_pred8 = kernel(area_pred8, pred_area)

    # intersection
    intersection =  K_targ1 + \
                    K_targ2 + \
                    K_targ3 + \
                    K_targ4 + \
                    K_targ5 + \
                    K_targ6 + \
                    K_targ7 + \
                    K_targ8

    # union
    # union = K_targ1 + K_pred1 - K_targ1*K_pred1 + \
    #         K_targ2 + K_pred2 - K_targ2*K_pred2 + \
    #         K_targ3 + K_pred3 - K_targ3*K_pred3 + \
    #         K_targ4 + K_pred4 - K_targ4*K_pred4 + \
    #         K_targ5 + K_pred5 - K_targ5*K_pred5 + \
    #         K_targ6 + K_pred6 - K_targ6*K_pred6 + \
    #         K_targ7 + K_pred7 - K_targ7*K_pred7 + \
    #         K_targ8 + K_pred8 - K_targ8*K_pred8 + \
    #         eps

    # calculate iou and loss
    ious = torch.abs(intersection / 8)
    loss = 1 - ious #-ious.log()

    return loss

def kernel(pred, target, k=4):
    assert pred.shape == target.shape
    x = torch.abs( (pred - target) / target )
    kf = 2 / ( 1 + torch.exp(k * x) )
    return kf

# def kernel(pred, target, k=-10):
#     x = torch.abs( (pred - target) / target )
#     sigmoid = 1 / ( 1 + torch.exp(k * (x)) )
#     kf = 2 * (1 - sigmoid)
#     return kf

def rectangle_area(bbox, bpts):
    assert bbox.shape[1] == 8
    area_1 = triangle_area(bbox[:, 2:4], bbox[:, 0:2], bpts)
    area_2 = triangle_area(bbox[:, 0:2], bbox[:, 4:6], bpts)
    area_3 = triangle_area(bbox[:, 4:6], bbox[:, 6:8], bpts)
    area_4 = triangle_area(bbox[:, 6:8], bbox[:, 2:4], bpts)
    area = area_1 + area_2 + area_3 + area_4
    return area

def poly_area(poly):
    assert poly.shape[1] == 8
    area_1 = triangle_area(poly[:, 0:2], poly[:, 2:4], poly[:, 4:6])
    area_2 = triangle_area(poly[:, 2:4], poly[:, 6:8], poly[:, 4:6])
    area = area_1 + area_2
    
    return area

def triangle_area(point_one, point_two, bpts_point):
    assert point_one.shape[1] == 2
    assert point_two.shape[1] == 2
    assert bpts_point.shape[1] == 2
    part_1 = point_one[:, 0]*bpts_point[:, 1] + bpts_point[:, 0]*point_two[:, 1] + point_two[:, 0]*point_one[:, 1]
    part_2 = point_one[:, 1]*bpts_point[:, 0] + bpts_point[:, 1]*point_two[:, 0] + point_two[:, 1]*point_one[:, 0]
    
    area = torch.abs((part_1 - part_2)/2)
    return area

@LOSSES.register_module()
class BoxPointsLoss(nn.Module):

    def __init__(self,
                 eps=1e-6, 
                 reduction='mean',
                 loss_weight=1.0):
        super(BoxPointsLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                bpts,
                # bbox,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (bpts * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape[0] == bpts.shape[0]
            weight = weight.mean(-1)
        loss = self.loss_weight * box_points_loss(
            bpts,
            # bbox,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        return loss
