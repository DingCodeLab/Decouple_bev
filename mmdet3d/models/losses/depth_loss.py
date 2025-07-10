import torch
from torch.cuda.amp.autocast_mode import autocast 
import torch.nn.functional as F 

import numpy as np
np.set_printoptions(threshold=np.inf)

def get_depth_loss(depth_labels, depth_preds, dbound):
    depth_channels = depth_preds.shape[1]  #。
    downsample_factor = depth_labels.shape[-1] // depth_preds.shape[-1]  #。

    depth_labels = get_downsampled_gt_depth(depth_labels, downsample_factor, depth_channels, dbound)  #。
    depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, depth_channels)  #。
    fg_mask = torch.max(depth_labels, dim=1).values > 0.0  #。
    
    # depth_preds_temp = depth_preds[fg_mask]
    # depth_labels_temp = depth_labels[fg_mask]
    
    # print(depth_labels_temp[:10,:].detach().cpu().numpy())
    # print(depth_preds_temp[:10,:].detach().cpu().numpy())

    with autocast(enabled=False):  
        depth_loss = (F.binary_cross_entropy(
            depth_preds[fg_mask],
            depth_labels[fg_mask],
            reduction='none',
        ).sum() / max(1.0, fg_mask.sum()))

    return depth_loss 

def get_downsampled_gt_depth(gt_depths, downsample_factor, depth_channels, dbound):
    """
    Input:
        gt_depths: [B, N, H, W]
    Output:
        gt_depths: [B*N*h*w, d]
    """
    B, N, H, W = gt_depths.shape  
    gt_depths = gt_depths.view(
        B * N, H // downsample_factor, downsample_factor, W // downsample_factor, downsample_factor, 1,
    ) 
    gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()  #。
    gt_depths = gt_depths.view(-1, downsample_factor * downsample_factor)  #。
    
    ##torch.where(a>0,a,b) a,b
    gt_depths_tmp = torch.where(gt_depths == 0.0,1e5 * torch.ones_like(gt_depths),gt_depths) 
    
    gt_depths = torch.min(gt_depths_tmp, dim=-1).values  #。
    
    # gt_depth_numpy = gt_depths.detach().cpu().numpy()[:30]
    # print(gt_depth_numpy)
    
    gt_depths = gt_depths.view(B * N, H // downsample_factor, W // downsample_factor)  #。

    gt_depths = (gt_depths - (dbound[0] - dbound[2])) / dbound[2]  #。
    
    #。
    gt_depths = torch.where((gt_depths < depth_channels + 1) & (gt_depths >= 0.0),gt_depths, torch.zeros_like(gt_depths))  
    gt_depths = F.one_hot(gt_depths.long(),num_classes=depth_channels + 1).view(-1, depth_channels + 1)[:, 1:]  #。
                          
    # gt_depth_numpy = gt_depths.detach().cpu().numpy()[:30,:]

    # print(gt_depth_numpy)

    return gt_depths.float()  #
