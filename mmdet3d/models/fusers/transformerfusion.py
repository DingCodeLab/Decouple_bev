# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.transformer import build_positional_encoding

from .mm_cross_attention import CustomMSDeformableAttention

from mmdet3d.models.builder import FUSERS
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16

import torch.nn.functional as F

from mmengine.visualization import Visualizer
import os

@FUSERS.register_module()
class PerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
    """

    def __init__(self,
                 camera_transformer=None,
                 lidar_transformer=None,
                 cqlkv_transformer=None,
                 lqckv_transformer=None,
                 fusion_transformer=None,
                 embed_dims=256,
                 lidar_dims=256,
                 camera_dims=256,
                 camera_positional_encoding=None,
                 lidar_positional_encoding=None,
                 bev_h=30,
                 bev_w=30,
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)
        self.camera_transformer = build_transformer_layer_sequence(camera_transformer)
        self.lidar_transformer = build_transformer_layer_sequence(lidar_transformer)
        self.cqlkv_transformer = build_transformer_layer_sequence(cqlkv_transformer)
        self.lqckv_transformer = build_transformer_layer_sequence(lqckv_transformer)
        self.fusion_transformer = build_transformer_layer_sequence(fusion_transformer)
        self.embed_dims = embed_dims
        self.lidar_dims = lidar_dims
        self.camera_dims = camera_dims
        self.fp16_enabled = False
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.camera_positional_encoding = build_positional_encoding(camera_positional_encoding)
        self.lidar_positional_encoding = build_positional_encoding(lidar_positional_encoding)

        self.init_layers()
        self.init_weights()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.lidar_bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.lidar_dims)
        self.camera_bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.camera_dims)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

    @auto_fp16(apply_to=('camera_feat','lidar_feat', 'camera_bev_queries','lidar_bev_queries', 'camera_bev_pos','lidar_bev_pos'))
    def get_bev_features(
            self,
            camera_feat,
            lidar_feat,
            camera_bev_queries,
            lidar_bev_queries,
            bev_h,
            bev_w,
            camera_bev_pos=None,
            lidar_bev_pos=None,
            **kwargs):
        """
        obtain bev features.
        """

        bs = lidar_feat.size(0)
        camera_bev_queries = camera_bev_queries.unsqueeze(1).repeat(1, bs, 1)
        lidar_bev_queries = lidar_bev_queries.unsqueeze(1).repeat(1, bs, 1)
        camera_bev_pos = camera_bev_pos.flatten(2).permute(2, 0, 1) # (bev_h*bev_w, bs, camera_dims)
        lidar_bev_pos = lidar_bev_pos.flatten(2).permute(2, 0, 1) # (bev_h*bev_w, bs, lidar_dims)

        spatial_shapes = []
        bs, _, h, w = camera_feat.shape
        spatial_shape = (h, w)
        spatial_shapes.append(spatial_shape)
        
        camera_feat = camera_feat.flatten(2).permute(0, 2, 1)
        lidar_feat = lidar_feat.flatten(2).permute(0, 2, 1)
        

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=lidar_bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        camera_bev_embed = self.camera_transformer(
            camera_bev_queries,
            camera_feat,
            camera_feat,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=camera_bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs
        )
        
        lidar_bev_embed = self.lidar_transformer(
            lidar_bev_queries,
            lidar_feat,
            lidar_feat,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=lidar_bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs
        )
        
            
        
        camera_bev_embed = self.cqlkv_transformer(
            camera_bev_embed.permute(1, 0, 2),
            camera_feat,
            lidar_feat,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=camera_bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs
        )
        
        lidar_bev_embed = self.lqckv_transformer(
            lidar_bev_embed.permute(1, 0, 2),
            lidar_feat,
            camera_feat,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=lidar_bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs
        )
        
        dual_bev_embed = torch.cat((camera_bev_embed,lidar_bev_embed), dim=-1).permute(1, 0, 2)
        dual_feat = torch.cat((camera_feat,lidar_feat), dim=-1)
        dual_bev_pos = torch.cat((camera_bev_pos,lidar_bev_pos), dim=-1)


        bev_embed = self.fusion_transformer(
            dual_bev_embed,
            dual_feat,
            dual_feat,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=dual_bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs
        )    

        return bev_embed
    
    
    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (B, C, H, W).
 
        Returns:
        """
        # mlvl_feats = torch.cat((mlvl_feats[0],mlvl_feats[1]), dim=1)
        
        camera_feats = mlvl_feats[0]
        lidar_feats = mlvl_feats[1]
        
        bs, _, _, _ = lidar_feats.shape
        dtype = lidar_feats.dtype
        
        camera_bev_queries = self.camera_bev_embedding.weight.to(dtype)
        lidar_bev_queries = self.lidar_bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=lidar_bev_queries.device).to(dtype)
        camera_bev_pos = self.camera_positional_encoding(bev_mask).to(dtype)
        lidar_bev_pos = self.lidar_positional_encoding(bev_mask).to(dtype)

        outputs = self.get_bev_features(
                camera_feats,
                lidar_feats,
                camera_bev_queries,
                lidar_bev_queries,
                self.bev_h,
                self.bev_w,
                camera_bev_pos=camera_bev_pos,
                lidar_bev_pos=lidar_bev_pos,
        )
        return outputs
