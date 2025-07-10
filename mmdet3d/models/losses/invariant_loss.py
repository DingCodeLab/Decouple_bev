import torch
import torch.nn as nn
import torch.nn.functional as F

class InvaraintInfoNCE(nn.Module):
    # invar_dim: (B, invar_dim, H*W)
    def __init__(self, invar_dim, camera_bev_dim,lidar_bev_dim, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.camera_project = nn.Conv2d(camera_bev_dim,invar_dim, kernel_size=1, stride=1, padding=0)
        self.lidar_project = nn.Conv2d(lidar_bev_dim,invar_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, cam_invar, lidar_invar, cam_bev, lidar_bev):
        
        # cam_invar: (B, invar_dim, H,W)
        # lidar_invar: (B, invar_dim, H,W)
        # cam_bev: (B, camera_bev_dim, H,W)
        
        #
        cam_invar = F.adaptive_avg_pool2d(cam_invar, (cam_invar.shape[2]//4, cam_invar.shape[3]//4))
        lidar_invar = F.adaptive_avg_pool2d(lidar_invar, (lidar_invar.shape[2]//4, lidar_invar.shape[3]//4))
        cam_bev = F.adaptive_avg_pool2d(cam_bev, (cam_invar.shape[2], cam_invar.shape[3]))
        lidar_bev = F.adaptive_avg_pool2d(lidar_bev, (lidar_invar.shape[2], lidar_invar.shape[3]))
        
        # project
        cam_bev = self.camera_project(cam_bev)  # (B, invar_dim, H,W)
        lidar_bev = self.lidar_project(lidar_bev)  # (B, invar_dim, H,W)
        
        #InfoNC
        
        # flatten: (B, C, H, W) → (B, C, N)
        B, C, H, W = cam_invar.shape

        def flatten_and_norm(feat):
            feat = feat.view(B, C, -1)              # (B, C, N)
            feat = F.normalize(feat, dim=1)         # channel
            return feat

        cam_invar = flatten_and_norm(cam_invar)     # (B, C, N)
        lidar_invar = flatten_and_norm(lidar_invar)
        cam_bev = flatten_and_norm(cam_bev)
        lidar_bev = flatten_and_norm(lidar_bev)

        # reshape: (B, C, N) → (B*N, C)
        anchor = cam_invar.permute(0, 2, 1).reshape(-1, C)       # (B*N, C)
        positive = lidar_invar.permute(0, 2, 1).reshape(-1, C)   # (B*N, C)
        neg1 = cam_bev.permute(0, 2, 1).reshape(-1, C)
        neg2 = lidar_bev.permute(0, 2, 1).reshape(-1, C)

        # anchor positive[i] neg1[i], neg2[i]
        pos_sim = (anchor * positive).sum(dim=1, keepdim=True) / self.temperature  # (B*N, 1)
        neg_sim1 = (anchor * neg1).sum(dim=1, keepdim=True) / self.temperature
        neg_sim2 = (anchor * neg2).sum(dim=1, keepdim=True) / self.temperature

        logits = torch.cat([pos_sim, neg_sim1, neg_sim2], dim=1)   # (B*N, 3)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss


class InvariantLoss(nn.Module):
    def __init__(self, invar_dim, camera_bev_dim,lidar_bev_dim):
        super().__init__()
        self.camera_project = nn.Conv2d(camera_bev_dim,invar_dim, kernel_size=1, stride=1, padding=0)
        self.lidar_project = nn.Conv2d(lidar_bev_dim,invar_dim, kernel_size=1, stride=1, padding=0)

    def sim_loss(self, cam_feat, lidar_feat):
        # L2
        return F.mse_loss(cam_feat, lidar_feat)

    def orth_loss(self, invar_feat, bev_feat):
        # invar bev
        B, C, H, W = invar_feat.shape
        invar = F.normalize(invar_feat.view(B, C, -1), dim=1)   # (B, C, N)
        bev = F.normalize(bev_feat.view(B, C, -1), dim=1)       # (B, C, N)
        dot = (invar * bev).sum(dim=1).mean()                   # (B, N) → scalar
        return dot

    def forward(self, cam_invar, lidar_invar, cam_bev, lidar_bev):
        # cam_invar: (B, invar_dim, H,W)
        # lidar_invar: (B, invar_dim, H,W)
        # cam_bev: (B, camera_bev_dim, H,W)
        
        #
        cam_invar = F.adaptive_avg_pool2d(cam_invar, (cam_invar.shape[2]//4, cam_invar.shape[3]//4))
        lidar_invar = F.adaptive_avg_pool2d(lidar_invar, (lidar_invar.shape[2]//4, lidar_invar.shape[3]//4))
        cam_bev = F.adaptive_avg_pool2d(cam_bev, (cam_invar.shape[2], cam_invar.shape[3]))
        lidar_bev = F.adaptive_avg_pool2d(lidar_bev, (lidar_invar.shape[2], lidar_invar.shape[3]))
        
        # project
        cam_bev = self.camera_project(cam_bev)  # (B, invar_dim, H,W)
        lidar_bev = self.lidar_project(lidar_bev)  # (B, invar_dim, H,W)
        
        loss_sim = self.sim_loss(cam_invar, lidar_invar)
        
        loss_orth = torch.abs(self.orth_loss(cam_invar, cam_bev)) + torch.abs(self.orth_loss(lidar_invar, lidar_bev))
        
        return 10* (loss_sim + loss_orth)
