import torch

__all__ = ["bev_ramdom_mask","bev_feat_add_gaussian_noise","bev_feat_selective_gaussian_noise"]

def create_mask(size, block_size, p):
    """
    创建一个指定大小的遮罩，其中的小块(block)按照给定概率p置1，其他置0。
    
    :param size: tuple(int, int), 遮罩的尺寸(height, width)
    :param block_size: int, 每个小块的尺寸
    :param p: float, 小块中元素被设置为1的概率
    :return: Tensor, 生成的遮罩
    """
    # # 计算每个维度上的小块数量
    # blocks_per_dim0 = size[0] // block_size
    # blocks_per_dim1 = size[1] // block_size
    
    # 计算每个维度上的小块数量
    blocks_per_dim0 = size[0] // block_size
    blocks_per_dim1 = size[1] // block_size
    total_blocks = blocks_per_dim0 * blocks_per_dim1

    # 计算要激活的块的数量
    num_active_blocks = int(p * total_blocks)

    # 创建一维遮罩数组，恰好 num_active_blocks 个值为1
    block_mask = torch.zeros(total_blocks)
    block_mask[:num_active_blocks] = 1
    
    # 使用 torch.randperm 生成一个随机排列的索引，用以打乱 block_mask
    perm = torch.randperm(total_blocks)
    
    # 将一维遮罩重构为二维形状
    block_mask = block_mask[perm].view(blocks_per_dim0, blocks_per_dim1)
    
    
    # 将随机块扩展到全尺寸遮罩
    mask = torch.repeat_interleave(block_mask, block_size, dim=0)
    mask = torch.repeat_interleave(mask, block_size, dim=1)
    
    # # 如果尺寸不完全匹配，调整遮罩尺寸
    # if mask.shape[0] != size[0] or mask.shape[1] != size[1]:
    #     mask = mask[:size[0], :size[1]]
    
    return mask.float()

def bev_ramdom_mask(features,block_size=9,probability=0.25):
    
    batch_size, channels ,height, width = features.shape

    # 创建遮罩
    mask = create_mask((height, width), block_size, probability)
    expanded_mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, -1, -1)
    masked_features = features * expanded_mask.to(features.device)
    return masked_features


def bev_feat_add_gaussian_noise(features, var_ratio=0.1, noise_percentage=0.25):
    # 计算特征图的均值、最大值和最小值
    mean_value = torch.mean(features)
    max_value = torch.max(features)
    min_value = torch.min(features)
    
    # 计算方差
    ref_value = torch.min(max_value - mean_value, mean_value - min_value)
    variance = ref_value * var_ratio
    
    # 不同通道 生成掩码，随机选择 25% 的像素点
    # mask = torch.rand(features.shape, device=features.device) <= noise_percentage
    
    # # 同通道  生成掩码，随机选择 25% 的像素点
    mask_size = features.shape[-2:]  # 获取特征图的后两维度的大小
    mask = torch.rand(mask_size, device=features.device) <= noise_percentage  # 生成掩码张量
    mask = mask.repeat(features.shape[0], features.shape[1], 1, 1)
    
    
    # 生成高斯噪声并应用到选中的像素点
    noise = torch.randn(features.shape, device=features.device) * torch.sqrt(variance)
    
    
    # features = features + noise.detach()* mask.detach()
    
    features = features + noise * mask
    
    return features

def bev_feat_selective_gaussian_noise(features, var_ratio=0.1, noise_percentage=0.25):
    # 计算特征图的均值、最大值和最小值
    mean_value = torch.mean(features)
    max_value = torch.max(features)
    min_value = torch.min(features)
    
    # 计算方差
    ref_value = torch.min(max_value - mean_value, mean_value - min_value)
    variance = ref_value * var_ratio
    
    B,_,H,W = features.shape
    total_ones = int(noise_percentage * B * H * W)
    
    ## 用特征图生成空间注意力图 p
    ## mask每个位置为1的概率为空间注意力图p对应的值  即 mask ~ Bernoulli(p)
    p=torch.softmax(torch.mean(features,dim=1).view(B,H*W),dim=1).view(B,H,W)
    indices = torch.multinomial(p.view(-1), total_ones, replacement=False)
    
    # ## 可视化 存储 空间注意图
    # p = p.view(B,H,W).detach().cpu().numpy()
    # import matplotlib.pyplot as plt
    # plt.imshow(p[0], cmap='jet', aspect='auto')
    # plt.axis('off')  # 关闭坐标轴
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除留白
    # plt.savefig("spatial_attention_map.png", bbox_inches='tight', pad_inches=0)
    
    mask = torch.zeros( B * H * W, dtype=torch.float32, device=features.device)
    mask[indices] = 1.0
    mask = mask.view(B, H, W).unsqueeze(1).repeat(1, features.shape[1], 1, 1)
    
    
    # 生成高斯噪声并应用到选中的像素点 noise ~ N(0, variance)
    noise = torch.randn(features.shape, device=features.device) * torch.sqrt(variance)
    
    features = features + noise.detach()* mask.detach()
    
    # features = features + noise * mask
    
    return features