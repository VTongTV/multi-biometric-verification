
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from loguru import logger

# Add project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.ensemble.ensemble_model import BiometricEnsemble

def run_protection_demo():
    logger.info("="*60)
    logger.info("TEMPLATE PROTECTION DEMO: CHAFFING + BIOHASHING")
    logger.info("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Initialize Model (Random Weights is fine for this demo)
    logger.info("Initializing Ensemble Model...")
    model = BiometricEnsemble(device=device)
    model.eval()
    
    # 2. Create Fake Inputs (Simulate a user)
    # 1 user, 4 modalities
    batch_size = 1
    # Shapes: Face(3,112,112), Finger(3,224,224), Iris(3,224,224)
    # We just need to simulate the *embeddings* really, but let's go through forward
    # actually, to test protection we can just inject embeddings directly into the dict structure
    # to avoid needing image inputs.
    
    logger.info("Generating Synthetic Embeddings...")
    
    # User A (Genuine)
    user_a_raw = {
        'embeddings': {
            'face': F.normalize(torch.randn(1, 512, device=device), p=2, dim=1),
            'fingerprint': F.normalize(torch.randn(1, 512, device=device), p=2, dim=1),
            'iris_nir': F.normalize(torch.randn(1, 512, device=device), p=2, dim=1),
            'iris_vis': F.normalize(torch.randn(1, 512, device=device), p=2, dim=1),
        },
        'qualities': {
            'face': torch.tensor([1.0], device=device),
            'fingerprint': torch.tensor([1.0], device=device),
            'iris_nir': torch.tensor([1.0], device=device),
            'iris_vis': torch.tensor([1.0], device=device),
        }
    }
    
    # User B (Impostor)
    user_b_raw = {
        'embeddings': {
            'face': F.normalize(torch.randn(1, 512, device=device), p=2, dim=1),
            'fingerprint': F.normalize(torch.randn(1, 512, device=device), p=2, dim=1),
            'iris_nir': F.normalize(torch.randn(1, 512, device=device), p=2, dim=1),
            'iris_vis': F.normalize(torch.randn(1, 512, device=device), p=2, dim=1),
        },
        'qualities': user_a_raw['qualities']
    }
    
    # 3. Baseline Match (Raw vs Raw)
    score_base = model.compute_score(user_a_raw, user_a_raw)['fused_score'].item()
    logger.info(f"Baseline (Raw vs Raw) Self-Match Score: {score_base:.4f} (Expected: ~1.0)")
    
    # 4. Protected Match (Key A vs Key A) - CORRECT KEY
    key_a = "secret_password_123"
    
    user_a_prot_A = model.protect_embeddings(user_a_raw, key_a)
    user_a_prot_A_2 = model.protect_embeddings(user_a_raw, key_a) # Re-generate to ensure determinism
    
    score_prot_match = model.compute_score(user_a_prot_A, user_a_prot_A_2)['fused_score'].item()
    logger.info(f"Protected (Key A vs Key A) Self-Match Score: {score_prot_match:.4f} (Expected: ~1.0)")
    
    # 5. Protected Mismatch (Key A vs Key B) - WRONG KEY / UNLINKABILITY
    # Same user, different keys. Should NOT match.
    key_b = "different_password_456"
    user_a_prot_B = model.protect_embeddings(user_a_raw, key_b)
    
    score_prot_mismatch = model.compute_score(user_a_prot_A, user_a_prot_B)['fused_score'].item()
    logger.info(f"Unlinkability (Key A vs Key B) Score: {score_prot_mismatch:.4f} (Expected: ~0.0)")
    
    if score_prot_mismatch < 0.2:
        logger.success("[PASS] Unlinkability Verified: Different keys produce uncorrelated templates.")
    else:
        logger.error("[FAIL] Unlinkability Failed: Templates are still correlated!")

    # 6. Irreversibility (Protected vs Raw)
    # If attacker leaks Raw database, can they match against Protected probe?
    # Or vice versa.
    # Note: Dimensions might match (512), but content is projected.
    score_cross = model.compute_score(user_a_prot_A, user_a_raw)['fused_score'].item()
    logger.info(f"Irreversibility (Protected vs Raw) Score: {score_cross:.4f} (Expected: ~0.0)")
    
    if score_cross < 0.2:
        logger.success("[PASS] Irreversibility Verified: Protected template does not match Raw template.")
    else:
        logger.error("[FAIL] Irreversibility Failed!")
        
    # 7. Impostor Attack (User A Key A vs User B Key A)
    # Attacker steals Key A, but has their own biometrics (User B).
    user_b_prot_A = model.protect_embeddings(user_b_raw, key_a)
    score_impostor = model.compute_score(user_a_prot_A, user_b_prot_A)['fused_score'].item()
    logger.info(f"Impostor (User A+Key A vs User B+Key A) Score: {score_impostor:.4f} (Expected: Low)")
    
    if score_impostor < 0.4:
         logger.success("[PASS] Security Verified: Stolen key is useless without correct biometrics.")
    else:
         logger.warning(f"[WARNING] Impostor Score is high: {score_impostor}. Biometric discrimination might be preserved too well?")

if __name__ == "__main__":
    run_protection_demo()
