
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
from loguru import logger

from ..encoders.biometric_encoder import BiometricEncoder
from ..fingerprint.ridge_attention import RidgeAttention
from .template_protection import TemplateProtector

class FingerprintEncoderWithAttention(nn.Module):
    """Wrapper that properly integrates Ridge Attention into the encoder forward pass."""
    
    def __init__(self, encoder, ridge_attention):
        super().__init__()
        self.encoder = encoder
        self.ridge_attention = ridge_attention
        # Expose these for weight loading if needed
        self.embedding_head = encoder.embedding_head
        self.backbone = encoder.backbone
        
        # Disable encoder's built-in attention to prevent double attention
        if hasattr(encoder, 'use_attention'):
            encoder.use_attention = False
    
    def forward(self, x, normalize=True):
        # Extract backbone features
        features = self.encoder.backbone.features(x)
        
        # Apply ridge-specific attention
        features = self.ridge_attention(features)
        
        # Clamp for FP16 stability (matching training)
        features = torch.clamp(features, min=-65000, max=65000)
        
        # Pool and embed
        features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        features = features.float()  # Ensure float32 before embedding head
        embeddings = self.encoder.embedding_head(features)
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

class BiometricEnsemble(nn.Module):
    """
    Unified Biometric Ensemble Model.
    
    Implements STRICT MODALITY PARTITIONING:
    - Independent sub-networks for Face, Fingerprint, and TWO Iris domains.
    - Zero crosstalk between modalities during feature extraction.
    - Quality-Aware Score Level Fusion.
    - Dynamic handling of missing modalities via Active Flags.
    """
    
    # Quality thresholds (derived from training norms)
    FACE_QUALITY_THRESHOLD = 12.0
    FINGERPRINT_QUALITY_THRESHOLD = 1.5
    IRIS_QUALITY_THRESHOLD = 10.0
    
    def __init__(
        self,
        face_weights_path: Optional[str] = None,
        iris_nir_weights_path: Optional[str] = None,
        iris_vis_weights_path: Optional[str] = None,
        fingerprint_weights_path: Optional[str] = None,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        
        # Register Active Flags (buffers) to track which modalities are valid
        self.register_buffer('face_active', torch.tensor(False))
        self.register_buffer('fingerprint_active', torch.tensor(False))
        self.register_buffer('iris_nir_active', torch.tensor(False))
        self.register_buffer('iris_vis_active', torch.tensor(False))
        
        # --- 1. Face Branch (Independent) ---
        logger.info("Initializing Face Branch...")
        
        # Default initialization (EfficientNet)
        backbone_type = 'iresnet50'
        pretrained_weights = None
        use_pretrained = True
        
        # Intelligent Backbone Detection
        if face_weights_path:
            path_str = str(face_weights_path)
            if path_str.endswith('.onnx'):
                # Explicit ONNX file
                backbone_type = 'iresnet50'
                pretrained_weights = path_str
            else:
                # Checkpoint file - inspect keys to determine backbone
                try:
                    chk = torch.load(face_weights_path, map_location='cpu')
                    state = chk.get('encoder_state_dict', chk)
                    keys = list(state.keys())
                    
                    # Check for ONNX-specific keys
                    is_onnx = any('initializers.onnx_initializer' in k for k in keys)
                    if is_onnx:
                        logger.info("Detected ONNX-based IResNet backbone in face checkpoint.")
                        backbone_type = 'iresnet50'
                        # We need the original ONNX file to construct the backbone skeleton
                        # Assuming standard location relative to project root
                        onnx_path = Path('weights/R50_Glint360K.onnx')
                        if onnx_path.exists():
                            pretrained_weights = str(onnx_path)
                        else:
                            # Fallback: try to find it relative to current file if needed, or assume global path
                            # For now, just warn. BiometricEncoder might fail if pretrained_weights not found
                            logger.warning(f"ONNX skeleton required but not found at {onnx_path}. Face model loading may fail.")
                    else:
                        logger.info("Detected EfficientNet backbone in face checkpoint.")
                        use_pretrained = False # We will load custom weights
                        
                except Exception as e:
                    logger.warning(f"Failed to inspect face checkpoint: {e}. Defaulting to EfficientNet.")

        self.face_encoder = BiometricEncoder(
            embedding_dim=512,
            backbone_type=backbone_type,
            pretrained=use_pretrained,
            pretrained_weights_path=pretrained_weights if pretrained_weights else ''
        )
        
        logger.info(f"Face encoder backbone type: {backbone_type}, pretrained_weights_path: {pretrained_weights}")
        
        if face_weights_path:
            # If it was an ONNX file, BiometricEncoder already loaded it via pretrained_weights_path
            # If it was a checkpoint, we load it now
            if not str(face_weights_path).endswith('.onnx'):
                if self._load_state_dict_safe(self.face_encoder, face_weights_path, "Face"):
                    self.face_active.fill_(True)
            else:
                # ONNX load assumed successful if init passed
                self.face_active.fill_(True)
            
            logger.info(f"Face branch initialized. face_active: {self.face_active}")

        # --- 2. Iris NIR Branch (Independent) ---
        logger.info("Initializing Iris NIR Branch (CASIA/Standard)...")
        self.iris_nir_encoder = BiometricEncoder(
            embedding_dim=512,
            backbone_type='efficientnet',
            use_attention=True,
            pretrained=True
        )
        if iris_nir_weights_path:
            if self._load_state_dict_safe(self.iris_nir_encoder, iris_nir_weights_path, "Iris NIR"):
                self.iris_nir_active.fill_(True)
                
        # --- 3. Iris Visible Branch (Independent) ---
        logger.info("Initializing Iris Visible Branch (UBIRIS/Mobile)...")
        self.iris_vis_encoder = BiometricEncoder(
            embedding_dim=512,
            backbone_type='efficientnet',
            use_attention=True,
            pretrained=True
        )
        if iris_vis_weights_path:
            if self._load_state_dict_safe(self.iris_vis_encoder, iris_vis_weights_path, "Iris Visible"):
                self.iris_vis_active.fill_(True)
        
        # --- 4. Fingerprint Branch (Independent) ---
        logger.info("Initializing Fingerprint Branch...")
        # Fingerprint uses specific wrapper and attention
        base_encoder = BiometricEncoder(
            embedding_dim=512,
            in_channels=3,
            backbone_type='efficientnet',
            pretrained=True,
            use_attention=False # Disable standard attention, use RidgeAttention
        )
        ridge_attn = RidgeAttention(channels=1280, reduction=16)
        self.fingerprint_encoder = FingerprintEncoderWithAttention(base_encoder, ridge_attn)
        
        if fingerprint_weights_path:
            if self._load_state_dict_safe(self.fingerprint_encoder, fingerprint_weights_path, "Fingerprint"):
                self.fingerprint_active.fill_(True)

        # Freeze All Backbones
        for param in self.parameters():
            param.requires_grad = False
        
        # Fusion Parameters (Learnable but initialized equally)
        # 4 Weights: Face, IrisNIR, IrisVis, Fingerprint
        self.fusion_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        self.fusion_weights.requires_grad = True 
        
        # Template Protection Module
        self.protector = TemplateProtector(chaff_dim=512)
        
    def _load_state_dict_safe(self, model, path, name="Model"):
        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            if 'encoder_state_dict' in checkpoint:
                state_dict = checkpoint['encoder_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            new_state_dict = {}
            for k, v in state_dict.items():
                # Clean up prefixes
                key = k.replace('module.', '')
                # If loading into wrapper, prefixes might need adjustment
                # But typically we load what matches. 
                # For fingerprint wrapper, the keys should match the wrapper structure 
                # (encoder.backbone... etc) or if training was done on wrapper.
                new_state_dict[key] = v
                
            model.load_state_dict(new_state_dict, strict=False)
            logger.info(f"Loaded {name} weights")
            return True
        except Exception as e:
            logger.warning(f"Could not load {name} weights: {e}")
            return False

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Forward pass with support for 4 modalities.
        Input keys: 'face', 'fingerprint', 'iris_nir', 'iris_vis'.
        Legacy key 'iris' maps to 'iris_nir'.
        """
        batch_size = 0
        device = self.device
        
        # Check available inputs
        all_keys = ['face', 'fingerprint', 'iris_nir', 'iris_vis', 'iris']
        for key in all_keys:
            if inputs.get(key) is not None:
                batch_size = inputs[key].shape[0]
                device = inputs[key].device
                break
        
        if batch_size == 0:
            return {'fused_score': torch.tensor(0.0)}

        embeddings = {}
        qualities = {}
        
        # --- Face ---
        if inputs.get('face') is not None and self.face_active:
            data = inputs['face'].to(device)
            # Extract unnormalized for quality, then normalize
            feats = self.face_encoder(data, normalize=False)
            norm = torch.norm(feats, p=2, dim=1)
            logger.info(f"Face Norm: {norm.mean().item():.4f} (threshold: {self.FACE_QUALITY_THRESHOLD})")
            q = torch.clamp(norm / self.FACE_QUALITY_THRESHOLD, 0.0, 1.0)
            logger.info(f"Face Quality: {q.mean().item():.4f}")
            embeddings['face'] = F.normalize(feats, p=2, dim=1)
            qualities['face'] = q
            
        # --- Fingerprint ---
        if inputs.get('fingerprint') is not None and self.fingerprint_active:
            data = inputs['fingerprint'].to(device)
            # Wrapper supports normalize=False
            feats = self.fingerprint_encoder(data, normalize=False)
            norm = torch.norm(feats, p=2, dim=1)
            logger.info(f"Fingerprint Norm: {norm.mean().item():.4f} (threshold: {self.FINGERPRINT_QUALITY_THRESHOLD})")
            # Use raw norm ratio (don't subtract 1.0 to avoid 0 quality for normalized embeddings)
            q = torch.clamp(norm / self.FINGERPRINT_QUALITY_THRESHOLD, 0.0, 1.0)
            logger.info(f"Fingerprint Quality: {q.mean().item():.4f}")
            embeddings['fingerprint'] = F.normalize(feats, p=2, dim=1)
            qualities['fingerprint'] = q
            
        # --- Iris NIR (Standard) ---
        # Handle 'iris' key as 'iris_nir' for backward compatibility
        nir_input = inputs.get('iris_nir') if inputs.get('iris_nir') is not None else inputs.get('iris')
        
        if nir_input is not None and self.iris_nir_active:
            data = nir_input.to(device)
            feats = self.iris_nir_encoder(data, normalize=False)
            norm = torch.norm(feats, p=2, dim=1)
            logger.info(f"Iris NIR Norm: {norm.mean().item():.4f} (threshold: {self.IRIS_QUALITY_THRESHOLD})")
            q = torch.clamp(norm / self.IRIS_QUALITY_THRESHOLD, 0.0, 1.0)
            logger.info(f"Iris NIR Quality: {q.mean().item():.4f}")
            embeddings['iris_nir'] = F.normalize(feats, p=2, dim=1)
            qualities['iris_nir'] = q
            
        # --- Iris Visible (Mobile) ---
        if inputs.get('iris_vis') is not None and self.iris_vis_active:
            data = inputs['iris_vis'].to(device)
            feats = self.iris_vis_encoder(data, normalize=False)
            norm = torch.norm(feats, p=2, dim=1)
            logger.info(f"Iris VIS Norm: {norm.mean().item():.4f} (threshold: {self.IRIS_QUALITY_THRESHOLD})")
            q = torch.clamp(norm / self.IRIS_QUALITY_THRESHOLD, 0.0, 1.0)
            logger.info(f"Iris VIS Quality: {q.mean().item():.4f}")
            embeddings['iris_vis'] = F.normalize(feats, p=2, dim=1)
            qualities['iris_vis'] = q

        return {
            'embeddings': embeddings,
            'qualities': qualities
        }

    def protect_embeddings(self, output_dict: Dict[str, Any], key: str) -> Dict[str, Any]:
        """
        Apply Template Protection (Chaffing + BioHashing) to a dictionary of embeddings.
        Returns a NEW dictionary with protected embeddings.
        """
        if not key:
            return output_dict
            
        protected_embeddings = {}
        raw_embeddings = output_dict['embeddings']
        
        for modality, emb in raw_embeddings.items():
            # Apply protection
            # emb is [Batch, Dim]
            p_emb = self.protector.protect(emb, key)
            protected_embeddings[modality] = p_emb
            
        # Return new structure (preserving qualities, etc.)
        return {
            'embeddings': protected_embeddings,
            'qualities': output_dict['qualities'] # Qualities remain the same
        }

    def compute_score(self, probe_out: Dict[str, Any], gallery_out: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute similarity score between pre-computed embeddings.
        """
        p_emb = probe_out['embeddings']
        p_qual = probe_out['qualities']
        
        g_emb = gallery_out['embeddings']
        g_qual = gallery_out['qualities']
        
        scores = {}
        # Use first available key to determine batch size/device
        ref_key = list(p_emb.keys())[0] if p_emb else list(g_emb.keys())[0]
        device = p_emb[ref_key].device if p_emb else self.device
        
        final_score = torch.zeros(p_emb[ref_key].shape[0], device=device) if p_emb else torch.tensor(0.0, device=device)
        
        total_weight = torch.zeros_like(final_score)
        base_weights = F.softplus(self.fusion_weights)
        
        # Mapping: Name -> Index in fusion_weights
        modality_map = {
            'face': 0, 
            'iris_nir': 1, 
            'iris_vis': 2, 
            'fingerprint': 3
        }
        
        for key, idx in modality_map.items():
            if key in p_emb and key in g_emb:
                sim = F.cosine_similarity(p_emb[key], g_emb[key])
                scores[key] = sim
                logger.info(f"Similarity {key}: {sim.mean().item():.4f}")
                
                # Quality Weighting
                q = torch.min(p_qual[key], g_qual[key])
                w = base_weights[idx] * q
                logger.info(f"  Quality: {q.mean().item():.4f}, Weight: {w.mean().item():.4f}, BaseWeight: {base_weights[idx].item():.4f}")
                
                final_score += sim * w
                total_weight += w
        
        final_score = final_score / (total_weight + 1e-6)
        
        return {
            'fused_score': final_score,
            'individual_scores': scores,
            'weights': total_weight
        }

    def verify(self, probe_data: Dict[str, torch.Tensor], gallery_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Verify with dynamic fusion of up to 4 modalities.
        """
        probe_out = self(probe_data)
        gallery_out = self(gallery_data)
        return self.compute_score(probe_out, gallery_out)
