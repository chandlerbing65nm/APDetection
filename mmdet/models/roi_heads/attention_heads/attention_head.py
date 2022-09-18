# Copyright (c) OpenMMLab. All rights reserved.
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer

from mmdet.core import auto_fp16, force_fp32, obb_attention_target, bbox2type
from mmdet.models.builder import HEADS, build_loss

from mmcv.ops.carafe import CARAFEPack

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


@HEADS.register_module()
class AttentionHead(nn.Module):

    def __init__(self,
                 num_convs=2,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 num_classes=80,
                 head_count = 1,
                 bbox_type='obb',
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 loss_attention=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(AttentionHead, self).__init__()
        assert class_agnostic is True
        
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.head_count = head_count
        self.class_agnostic = class_agnostic

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.fp16_enabled = False
        self.loss_attention = build_loss(loss_attention)
        assert bbox_type in ['hbb', 'obb']
        self.bbox_type = bbox_type

        self.normalconvs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (self.in_channels if i == 0 else self.conv_out_channels)
            self.normalconvs.append(
                ConvModule(in_channels,
                            self.conv_out_channels,
                            3,
                            padding=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg = dict(type='ReLU')))
                    
        logits_in_channel = self.conv_out_channels
        out_channels = 1 if self.class_agnostic else self.num_classes
        self.conv_logits = nn.Conv2d(logits_in_channel, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        for m in [self.conv_logits]:
            if m is None:
                continue
            elif isinstance(m, CARAFEPack):
                m.init_weights()
            else:
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
        '''print("\nsize of Orig ROI: ", x.size())'''
        feat = x
        
        for conv3x3 in self.normalconvs:
            x = conv3x3(x)

        attention_pred = self.conv_logits(x)
        attention_feats = attention_pred * feat
        return attention_pred, attention_feats

    def get_targets(self, sampling_results, gt_attentions, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]

        attention_targets = obb_attention_target(
            pos_proposals, 
            pos_assigned_gt_inds, 
            gt_attentions, 
            rcnn_train_cfg, 
            self.bbox_type)

        return attention_targets

    @force_fp32(apply_to=('attention_pred', ))
    def loss(self, attention_pred, attention_targets, labels):
    
        loss = dict()
        if attention_pred.size(0) == 0:
            loss_attention = attention_pred.sum() * 0
            # loss_attention_2 = attention_pred[:, 1, :, :].unsqueeze(1).sum() * 0
        else:
            if self.class_agnostic:
                loss_attention = self.loss_attention(attention_pred, attention_targets, torch.zeros_like(labels))
            else:
                loss_attention = self.loss_attention(attention_pred, attention_targets, labels)
            
        loss['loss_att'] = loss_attention
        
        return loss