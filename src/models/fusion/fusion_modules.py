"""
Fusion modules for combining biometric embeddings.

Implements:
- Eye Fusion: Combines periocular and visible-iris embeddings
- TriModal Fusion: Combines face, eye, and fingerprint embeddings

Uses attention mechanisms and gated fusion for dynamic weighting.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class GatedFusion(nn.Module):
    """
    Gated fusion module for combining two embeddings.
    
    Uses learned gates to determine the contribution of each modality,
    allowing the network to dynamically weight inputs based on reliability.
    
    Args:
        embedding_dim: Dimension of input embeddings
        hidden_dim: Hidden dimension for gate computation
        output_dim: Output embedding dimension (default: same as input)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: Optional[int] = None,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim or embedding_dim
        
        # Gate network: determines weight for each modality
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Projection layer if output_dim != embedding_dim
        if self.output_dim != embedding_dim:
            self.projection = nn.Linear(embedding_dim, self.output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse two embeddings with learned gating.
        
        Args:
            emb1: First embedding [batch, embedding_dim]
            emb2: Second embedding [batch, embedding_dim]
            
        Returns:
            Fused embedding [batch, output_dim]
        """
        # Concatenate for gate input
        concat = torch.cat([emb1, emb2], dim=-1)
        
        # Compute gates
        gates = self.gate(concat)  # [batch, 2]
        g1, g2 = gates[:, 0:1], gates[:, 1:2]
        
        # Weighted combination
        fused = g1 * emb1 + g2 * emb2
        
        # Project and normalize
        output = self.projection(fused)
        output = F.normalize(output, p=2, dim=1)
        
        return output


class AttentionFusion(nn.Module):
    """
    Attention-based fusion for combining multiple embeddings.
    
    Uses self-attention to model cross-modal interactions before fusion.
    
    Args:
        embedding_dim: Dimension of input embeddings
        num_heads: Number of attention heads
        output_dim: Output embedding dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        num_heads: int = 8,
        output_dim: int = 512,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(embedding_dim)
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse embeddings with attention.
        
        Args:
            embeddings: Stacked embeddings [batch, num_modalities, embedding_dim]
            mask: Optional attention mask [batch, num_modalities]
            
        Returns:
            Fused embedding [batch, output_dim]
        """
        # Self-attention over modalities
        attended, _ = self.attention(
            embeddings, embeddings, embeddings,
            key_padding_mask=mask
        )
        
        # Residual connection and norm
        attended = self.norm(attended + embeddings)
        
        # Average pool over modalities
        fused = attended.mean(dim=1)
        
        # Project
        output = self.projection(fused)
        output = F.normalize(output, p=2, dim=1)
        
        return output


class EyeFusionModule(nn.Module):
    """
    Fusion module for combining periocular and visible-iris embeddings.
    
    Uses gated fusion with optional confidence weighting to handle
    cases where one modality may be unreliable.
    
    Args:
        embedding_dim: Input embedding dimension (default: 512)
        hidden_dim: Hidden dimension for gate network
        output_dim: Output embedding dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        
        # Gated fusion for periocular + iris
        self.gated_fusion = GatedFusion(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )
        
        # Confidence estimation networks
        self.periocular_confidence = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.iris_confidence = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        logger.info("EyeFusionModule initialized")
    
    def forward(
        self,
        periocular_emb: torch.Tensor,
        iris_emb: torch.Tensor,
        return_confidence: bool = False
    ) -> torch.Tensor:
        """
        Fuse periocular and iris embeddings.
        
        Args:
            periocular_emb: Periocular embedding [batch, embedding_dim]
            iris_emb: Visible-light iris embedding [batch, embedding_dim]
            return_confidence: Whether to return confidence scores
            
        Returns:
            Fused eye embedding [batch, output_dim]
            Optionally: tuple of (embedding, {periocular_conf, iris_conf})
        """
        # Estimate confidence for each modality
        peri_conf = self.periocular_confidence(periocular_emb)
        iris_conf = self.iris_confidence(iris_emb)
        
        # Weight embeddings by confidence before fusion
        weighted_peri = periocular_emb * peri_conf
        weighted_iris = iris_emb * iris_conf
        
        # Gated fusion
        fused = self.gated_fusion(weighted_peri, weighted_iris)
        
        if return_confidence:
            return fused, {
                'periocular_confidence': peri_conf,
                'iris_confidence': iris_conf
            }
        
        return fused


class TriModalFusionNet(nn.Module):
    """
    Trimodal fusion network for combining face, eye, and fingerprint.
    
    Architecture:
    1. Cross-modal attention to capture inter-modality relationships
    2. Gated fusion with residual connections
    3. Missing modality handling
    
    Args:
        embedding_dim: Input embedding dimension for each modality (default: 512)
        hidden_dim: Hidden dimension for fusion layers
        output_dim: Output embedding dimension
        num_attention_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_attention_heads: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_modalities = 3  # face, eye, fingerprint
        
        # Modality-specific projections
        self.face_proj = nn.Linear(embedding_dim, hidden_dim)
        self.eye_proj = nn.Linear(embedding_dim, hidden_dim)
        self.finger_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # Learnable modality embeddings (like positional encoding)
        self.modality_embeddings = nn.Parameter(
            torch.randn(3, hidden_dim) * 0.02
        )
        
        # Cross-modal attention
        self.cross_attention = AttentionFusion(
            embedding_dim=hidden_dim,
            num_heads=num_attention_heads,
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # Final gated fusion
        self.final_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        # Default embedding for missing modalities
        self.register_buffer(
            'default_embedding',
            torch.zeros(1, embedding_dim)
        )
        
        logger.info(
            f"TriModalFusionNet initialized: "
            f"embedding_dim={embedding_dim}, output_dim={output_dim}"
        )
    
    def forward(
        self,
        face_emb: Optional[torch.Tensor] = None,
        eye_emb: Optional[torch.Tensor] = None,
        fingerprint_emb: Optional[torch.Tensor] = None,
        return_weights: bool = False
    ) -> torch.Tensor:
        """
        Fuse up to three modality embeddings.
        
        Args:
            face_emb: Face embedding [batch, embedding_dim]
            eye_emb: Eye (fused) embedding [batch, embedding_dim]
            fingerprint_emb: Fingerprint embedding [batch, embedding_dim]
            return_weights: Whether to return fusion weights
            
        Returns:
            Fused embedding [batch, output_dim]
        """
        # Determine batch size
        for emb in [face_emb, eye_emb, fingerprint_emb]:
            if emb is not None:
                batch_size = emb.shape[0]
                device = emb.device
                break
        else:
            raise ValueError("At least one modality must be provided")
        
        # Handle missing modalities with default embedding
        if face_emb is None:
            face_emb = self.default_embedding.expand(batch_size, -1).to(device)
        if eye_emb is None:
            eye_emb = self.default_embedding.expand(batch_size, -1).to(device)
        if fingerprint_emb is None:
            fingerprint_emb = self.default_embedding.expand(batch_size, -1).to(device)
        
        # Project each modality
        face_proj = self.face_proj(face_emb) + self.modality_embeddings[0]
        eye_proj = self.eye_proj(eye_emb) + self.modality_embeddings[1]
        finger_proj = self.finger_proj(fingerprint_emb) + self.modality_embeddings[2]
        
        # Stack for attention [batch, 3, hidden_dim]
        stacked = torch.stack([face_proj, eye_proj, finger_proj], dim=1)
        
        # Cross-modal attention
        attended = self.cross_attention(stacked)
        
        # Gated combination
        concat = torch.cat([face_proj, eye_proj, finger_proj], dim=-1)
        gates = self.final_gate(concat)  # [batch, 3]
        
        weighted = (
            gates[:, 0:1] * face_proj + 
            gates[:, 1:2] * eye_proj + 
            gates[:, 2:3] * finger_proj
        )
        
        # Combine with attention output
        combined = weighted + attended
        
        # Final projection
        output = self.output_projection(combined)
        output = F.normalize(output, p=2, dim=1)
        
        if return_weights:
            return output, {
                'face_weight': gates[:, 0],
                'eye_weight': gates[:, 1],
                'fingerprint_weight': gates[:, 2]
            }
        
        return output


class SimpleScoreFusion(nn.Module):
    """
    Simple score-level fusion for combining similarity scores.
    
    Used as a baseline or when embeddings are not available.
    
    Args:
        num_modalities: Number of similarity scores to fuse
        learnable: Whether weights are learnable
    """
    
    def __init__(
        self,
        num_modalities: int = 3,
        learnable: bool = True
    ):
        super().__init__()
        
        if learnable:
            self.weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)
        else:
            self.register_buffer(
                'weights',
                torch.ones(num_modalities) / num_modalities
            )
    
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Fuse similarity scores.
        
        Args:
            scores: Similarity scores [batch, num_modalities]
            
        Returns:
            Fused score [batch]
        """
        normalized_weights = F.softmax(self.weights, dim=0)
        return (scores * normalized_weights).sum(dim=1)


def create_fusion_module(
    fusion_type: str,
    embedding_dim: int = 512,
    **kwargs
) -> nn.Module:
    """
    Factory function to create fusion modules.
    
    Args:
        fusion_type: One of 'eye', 'trimodal', 'score'
        embedding_dim: Embedding dimension
        **kwargs: Additional module-specific arguments
        
    Returns:
        Fusion module instance
    """
    fusion_type = fusion_type.lower()
    
    if fusion_type == 'eye':
        return EyeFusionModule(
            embedding_dim=embedding_dim,
            hidden_dim=kwargs.get('hidden_dim', 512),
            output_dim=kwargs.get('output_dim', 512),
            dropout=kwargs.get('dropout', 0.2)
        )
    elif fusion_type == 'trimodal':
        return TriModalFusionNet(
            embedding_dim=embedding_dim,
            hidden_dim=kwargs.get('hidden_dim', 512),
            output_dim=kwargs.get('output_dim', 512),
            num_attention_heads=kwargs.get('num_attention_heads', 8),
            dropout=kwargs.get('dropout', 0.2)
        )
    elif fusion_type == 'score':
        return SimpleScoreFusion(
            num_modalities=kwargs.get('num_modalities', 3),
            learnable=kwargs.get('learnable', True)
        )
    else:
        raise ValueError(
            f"Unknown fusion type: {fusion_type}. "
            f"Choose from: eye, trimodal, score"
        )
