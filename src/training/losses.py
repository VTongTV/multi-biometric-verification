import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaFace(nn.Module):
    """
    AdaFace: Quality Adaptive Margin for Face Recognition.
    CVPR 2022 (SOTA).
    
    Link: https://arxiv.org/abs/2204.00964
    
    This loss function effectively down-weights the contribution of 
    low-quality images (unidentifiable/burry) to the gradient, 
    preventing them from confounding the model.
    """
    def __init__(self, embedding_size=256, class_num=1000, m=0.4, h=0.333, s=64., t_alpha=1.0):
        super(AdaFace, self).__init__()
        self.class_num = class_num
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, class_num))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

    def forward(self, embeddings, norms, label):
        """
        Embeddings: (B, D) - Normalized features
        Norms: (B,) - Feature norms (proxy for image quality)
        Label: (B,) - Class labels
        """
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embeddings, kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batch statistics
        with torch.no_grad():
            self.batch_mean = norms.mean() * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = norms.std() * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% of data will be within [-1, 1]
        margin_scaler = margin_scaler * self.h # 68% of data will be within [-0.333, 0.333]
        margin_scaler = torch.clip(margin_scaler, -1, 1) # clip scalar range to [-1, 1]
        
        # g_angle: adaptive margin
        # if image is high quality (large norm), margin is smaller (harder sample)
        # if image is low quality (small norm), margin is larger (easier sample)
        # This is counter-intuitive? No.
        # Wait, the paper says: "We approximate image quality with feature norm."
        # High norm = High quality.
        # AdaFace adds margin: m_adaptive = -margin_scaler * m
        # If quality is high (scaler > 0), margin becomes negative -> reduces total margin? 
        # Actually paper equation: cos(theta + m_adaptive)
        # AdaFace logic:
        # High quality -> Focus on hard samples -> Small adaptive margin? 
        # Low quality -> Don't force them too hard -> Large adaptive margin? 
        # Actually: AdaFace emphasizes high quality samples more.
        
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=embeddings.device)
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=embeddings.device)
        current_m = self.m + margin_scaler # (B,)

        # One-hot
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Angular margin (ArcFace)
        theta = cosine.acos()
        
        # Add adaptive margin ONLY to the ground truth class
        # We need to broadcast current_m to the right positions
        output = cosine # Placeholder
        
        # Efficient implementation
        # For GT class: cos(theta + m_adaptive)
        m_add = current_m.view(-1, 1) * one_hot
        theta_m = torch.clip(theta + m_add, min=self.eps, max=math.pi-self.eps)
        cosine_m = theta_m.cos()
        
        # Scaled by s
        # Output = s * (cos(theta+m) for GT, cos(theta) for others)
        output = cosine * (1.0 - one_hot) + cosine_m * one_hot
        output *= self.s
        
        return output

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output
