import numpy as np
import torch
from torch.nn.modules.utils import _pair


def obb_attention_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_attentions_list, cfg, bbox_type='hbb'):
    """ Compute attention target for positive proposals in multiple images.

    Args:
        pos_proposals_list (list[Tensor]): Positive proposals in multiple
            images.
        pos_assigned_gt_inds_list (list[Tensor]): Assigned GT indices for each
            positive proposals.
        gt_attentions_list (list[:obj:`BaseInstanceattentions`]): Ground truth attentions of
            each image.
        cfg (dict): Config dict that specifies the attention size.

    Returns:
        list[Tensor]: attention target of each image.
    """
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    btype_list = [bbox_type for _ in range(len(pos_proposals_list))]
    attention_targets = map(attention_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_attentions_list,
                       cfg_list, btype_list)
    attention_targets = list(attention_targets)
    if len(attention_targets) > 0:
        attention_targets = torch.cat(attention_targets)
    return attention_targets


def attention_target_single(pos_proposals, pos_assigned_gt_inds, gt_attentions, cfg, btype):
    """Compute attention target for each positive proposal in the image.

    Args:
        pos_proposals (Tensor): Positive proposals.
        pos_assigned_gt_inds (Tensor): Assigned GT inds of positive proposals.
        gt_attentions (:obj:`BaseInstanceattentions`): GT attentions in the format of Bitmap
            or Polygon.
        cfg (dict): Config dict that indicate the attention size.

    Returns:
        Tensor: attention target of each positive proposals in the image.
    """
    device = pos_proposals.device
    attention_size = _pair(cfg.attention_size)
    num_pos = pos_proposals.size(0)
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        maxh, maxw = gt_attentions.height, gt_attentions.width
        if btype == 'hbb':
            proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
            proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

        attention_targets = gt_attentions.crop_and_resize(
            proposals_np, attention_size, device=device,
            inds=pos_assigned_gt_inds).to_ndarray()

        attention_targets = torch.from_numpy(attention_targets).float().to(device)
    else:
        attention_targets = pos_proposals.new_zeros((0, ) + attention_size)

    return attention_targets
