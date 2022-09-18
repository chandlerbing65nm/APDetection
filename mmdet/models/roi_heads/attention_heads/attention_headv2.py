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
class AttentionHeadv2(nn.Module):

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
        super(AttentionHeadv2, self).__init__()
        assert class_agnostic is True
        assert head_count
        
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size

        #################################################
        self.in_channels = in_channels
        self.key_channels = in_channels // 2
        self.head_count = head_count
        self.value_channels = in_channels
        #################################################

        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.fp16_enabled = False
        self.loss_attention = build_loss(loss_attention)
        assert bbox_type in ['hbb', 'obb']
        self.bbox_type = bbox_type

        self.keys = ConvModule(self.in_channels,
                                self.key_channels,
                                1,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                act_cfg = dict(type='ReLU'))

        self.queries = ConvModule(self.in_channels,
                                self.key_channels,
                                1,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                act_cfg = dict(type='ReLU'))

        self.values = ConvModule(self.in_channels,
                                self.value_channels,
                                1,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                act_cfg = dict(type='ReLU'))

        self.reprojection = ConvModule(self.value_channels,
                                self.in_channels,
                                1,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                act_cfg = dict(type='ReLU'))
                    
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
        feats = x
        n, _, h, w = x.size()

        keys = self.keys(x).reshape((n, self.key_channels, h * w))
        queries = self.queries(x).reshape(n, self.key_channels, h * w)
        values = self.values(x).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            query = F.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + x

        attention_pred = self.conv_logits(reprojected_value)
        attention_feats = attention

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