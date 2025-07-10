import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import FUSERS

__all__ = ["MOEFusers"]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class Expert(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)

class Router(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1)  #
        )

    def forward(self, x):
        routing_logits = self.conv(x) # (B, 3, H, W)
        routing_weights = F.softmax(routing_logits, dim=1)  # softmax
        return routing_weights

@FUSERS.register_module()
class MOEFusers(nn.Module):
    def __init__(self, c_camera=80, c_lidar=256, c_out=336, H=180, W=180,use_entropy_loss=False):
        super().__init__()
        self.H = H
        self.W = W

        self.expert_camera = Expert(c_camera, c_out)
        self.expert_lidar = Expert(c_lidar, c_out)
        self.expert_fusion = Expert(c_camera + c_lidar, c_out)

        self.router = Router(c_camera + c_lidar)
        self.use_entropy_loss = use_entropy_loss 

    def forward(self, camera_feat, lidar_feat):
        # camera_feat: (B, HW, 80), lidar_feat: (B, HW, 256)
        B, HW, _ = camera_feat.shape
        camera_feat = camera_feat.permute(0, 2, 1).reshape(B, -1, self.H, self.W)  # (B, 80, H, W)
        lidar_feat = lidar_feat.permute(0, 2, 1).reshape(B, -1, self.H, self.W)    # (B, 256, H, W)

        feat_cam = self.expert_camera(camera_feat)
        feat_lidar = self.expert_lidar(lidar_feat)

        fusion_input = torch.cat([camera_feat, lidar_feat], dim=1)
        feat_fusion = self.expert_fusion(fusion_input)

        routing_weights = self.router(fusion_input)  # (B, 3, H, W)
        w1, w2, w3 = torch.chunk(routing_weights, 3, dim=1)

        fused_feat = w1 * feat_cam + w2 * feat_lidar + w3 * feat_fusion  # (B, 336, H, W)

        # reshape back to (B, HW, 336)
        fused_feat = fused_feat.flatten(2).permute(0, 2, 1)
        
        
        
        # ======== Entropy Loss (Stable) ========
        if self.use_entropy_loss:
            eps = 1e-6
            routing_weights = routing_weights.clamp(min=eps, max=1.0)  # log(0)
            entropy_loss = 10 * (- (routing_weights * torch.log(routing_weights)).sum(dim=1)).mean()
        else:
            entropy_loss = None

        # # ======== Optional KL Regularization (Encourage balanced experts) ========
        # with torch.no_grad():
        #     uniform = torch.full_like(routing_probs, 1.0 / 3)

        # kl_loss = F.kl_div(routing_probs.log(), uniform, reduction='batchmean')

        # # ======== Combine (lambda) ========
        # lambda_entropy = 1.0
        # lambda_kl = 0.01
        # entropy_loss = lambda_entropy * entropy_loss + lambda_kl * kl_loss
        
        return fused_feat,entropy_loss  # (B, HW, 336)
