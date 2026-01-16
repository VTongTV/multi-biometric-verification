"""
Iris Domain Configurations.
Defines datasets and parameters for domain-specific training.
"""

DOMAINS = {
    'nir_casia': {
        'name': 'Near-Infrared (CASIA)',
        'description': 'Clean, studio-quality NIR images. Low noise.',
        'datasets': [
            'CASIA_T', 
            'CASIA_I', 
            'CASIA_L', 
            'MMU'
        ],
        # NIR images are cleaner, so we can use standard augmentation
        'augmentation_strength': 'medium', 
        'expected_samples': 39000,
        # NIR-specific optimizations
        'config_overrides': {
            'augmentation_strength': 'medium',
            'arcface_margin': 0.40,  # Slightly higher margin for cleaner data
        },
    },
    
    'visible': {
        'name': 'Visible Light (UBIRIS/SBVPI)',
        'description': 'Noisy, unconstrained visible light images. High noise/reflections.',
        'datasets': [
            'UBIRIS', 
            'SBVPI'
        ],
        # Visible images need very heavy augmentation (noise, blur, occlusion, color)
        'augmentation_strength': 'very_heavy',
        'expected_samples': 7600,
        # Domain-specific config overrides for better visible light accuracy
        # These settings address the 5x smaller dataset and inherent optical challenges
        'config_overrides': {
            'epochs': 300,              # Much longer training for small dataset
            'lr': 8e-5,                 # Lower LR for small dataset stability
            'backbone_lr_mult': 0.5,    # More backbone adaptation for visible
            'dropout': 0.6,             # Higher regularization (up from 0.4)
            'weight_decay': 2e-3,       # Stronger weight decay
            'early_stop_patience': 1,  # Much more patience for small dataset
            'mixup_prob': 0.5,          # MixUp probability
            'mixup_alpha': 0.6,         # Stronger mixup blending
            'use_cutmix': True,         # Enable CutMix for small dataset
            'cutmix_prob': 0.4,         # CutMix probability
            'cutmix_alpha': 1.0,        # CutMix beta distribution alpha
            'arcface_margin': 0.25,     # Lower margin for harder/noisier data
            'arcface_scale': 35.0,      # Lower scale for small dataset
            'margin_warmup_epochs': 10, # Longer warmup for small dataset
            'margin_schedule': 'cosine', # Smoother margin increase
            'augmentation_strength': 'very_heavy',
            'use_subcenter_arcface': True,  # Enable SubCenter for visible variation
            'num_subcenters': 4,        # More subcenters for visible variation
            'min_samples_per_class': 5, # Only keep reliable classes
        },
    }
    
    # Excluded: CASIA_S (Synthetic) - as per plan
}

def get_domain_datasets(domain_name):
    """Get list of dataset names key for a given domain."""
    if domain_name not in DOMAINS:
        raise ValueError(f"Unknown domain: {domain_name}. Valid: {list(DOMAINS.keys())}")
    return DOMAINS[domain_name]['datasets']


def get_domain_config_overrides(domain_name):
    """Get config overrides for a specific domain."""
    if domain_name not in DOMAINS:
        return {}
    return DOMAINS[domain_name].get('config_overrides', {})


def get_augmentation_strength(domain_name):
    """Get augmentation strength for a specific domain."""
    if domain_name not in DOMAINS:
        return 'standard'
    return DOMAINS[domain_name].get('augmentation_strength', 'standard')
