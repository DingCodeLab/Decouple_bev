import torch

__all__ = ["bev_ramdom_mask","bev_feat_add_gaussian_noise","bev_feat_selective_gaussian_noise"]

def create_mask(size, block_size, p):
    """
   (block10。
    
    :param size: tuple(int, int),(height, width)
    :param block_size: int,
    :param p: float,
    :return: Tensor,
    """
    # #
    # blocks_per_dim0 = size[0] // block_size
    # blocks_per_dim1 = size[1] // block_size
    
    #
    blocks_per_dim0 = size[0] // block_size
    blocks_per_dim1 = size[1] // block_size
    total_blocks = blocks_per_dim0 * blocks_per_dim1

    #
    num_active_blocks = int(p * total_blocks)

    # num_active_blocks1
    block_mask = torch.zeros(total_blocks)
    block_mask[:num_active_blocks] = 1
    
    # torch.randperm block_mask
    perm = torch.randperm(total_blocks)
    
    #
    block_mask = block_mask[perm].view(blocks_per_dim0, blocks_per_dim1)
    
    
    #
    mask = torch.repeat_interleave(block_mask, block_size, dim=0)
    mask = torch.repeat_interleave(mask, block_size, dim=1)
    
    # #
    # if mask.shape[0] != size[0] or mask.shape[1] != size[1]:
    #     mask = mask[:size[0], :size[1]]
    
    return mask.float()

def bev_ramdom_mask(features,block_size=9,probability=0.25):
    
    batch_size, channels ,height, width = features.shape

    #
    mask = create_mask((height, width), block_size, probability)
    expanded_mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, -1, -1)
    masked_features = features * expanded_mask.to(features.device)
    return masked_features


def bev_feat_add_gaussian_noise(features, var_ratio=0.1, noise_percentage=0.25):
    #
    mean_value = torch.mean(features)
    max_value = torch.max(features)
    min_value = torch.min(features)
    
    #
    ref_value = torch.min(max_value - mean_value, mean_value - min_value)
    variance = ref_value * var_ratio
    
    # 25%
    # mask = torch.rand(features.shape, device=features.device) <= noise_percentage
    
    # #  25%
    mask_size = features.shape[-2:]  #
    mask = torch.rand(mask_size, device=features.device) <= noise_percentage  #
    mask = mask.repeat(features.shape[0], features.shape[1], 1, 1)
    
    
    #
    noise = torch.randn(features.shape, device=features.device) * torch.sqrt(variance)
    
    
    # features = features + noise.detach()* mask.detach()
    
    features = features + noise * mask
    
    return features

def bev_feat_selective_gaussian_noise(features, var_ratio=0.1, noise_percentage=0.25):
    #
    mean_value = torch.mean(features)
    max_value = torch.max(features)
    min_value = torch.min(features)
    
    #
    ref_value = torch.min(max_value - mean_value, mean_value - min_value)
    variance = ref_value * var_ratio
    
    B,_,H,W = features.shape
    total_ones = int(noise_percentage * B * H * W)
    
    ## p
    ## mas  mask ~ Bernoulli(p)
    p=torch.softmax(torch.mean(features,dim=1).view(B,H*W),dim=1).view(B,H,W)
    indices = torch.multinomial(p.view(-1), total_ones, replacement=False)
    
    # ##
    # p = p.view(B,H,W).detach().cpu().numpy()
    # import matplotlib.pyplot as plt
    # plt.imshow(p[0], cmap='jet', aspect='auto')
    # plt.axis('off')  #
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  #
    # plt.savefig("spatial_attention_map.png", bbox_inches='tight', pad_inches=0)
    
    mask = torch.zeros( B * H * W, dtype=torch.float32, device=features.device)
    mask[indices] = 1.0
    mask = mask.view(B, H, W).unsqueeze(1).repeat(1, features.shape[1], 1, 1)
    
    
    # noise ~ N(0, variance)
    noise = torch.randn(features.shape, device=features.device) * torch.sqrt(variance)
    
    features = features + noise.detach()* mask.detach()
    
    # features = features + noise * mask
    
    return features