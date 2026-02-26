
import sys
import os
from pathlib import Path
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "webapp" / "backend"))

from src.api.core.config import DATA_DIR, GALLERY_DIR

def check_user_59():
    user_id = "user_59"
    gallery_path = GALLERY_DIR / user_id
    
    if not gallery_path.exists():
        print(f"Gallery path for {user_id} not found: {gallery_path}")
        return

    print(f"Checking data for {user_id} at {gallery_path}")
    
    # Load raw cache
    raw_dir = DATA_DIR / "raw"
    raw_cache = {}
    
    # Iris NIR
    nir_src = raw_dir / "CASIA-Iris-Interval"
    if nir_src.exists():
        raw_cache['irisNir'] = list(nir_src.glob("**/*.jpg"))
        print(f"Iris NIR raw count: {len(raw_cache['irisNir'])}")
    
    # Iris Vis
    vis_src = raw_dir / "ubiris"
    if vis_src.exists():
        raw_cache['irisVis'] = list(vis_src.glob("**/*.tiff"))
        print(f"Iris Vis raw count: {len(raw_cache['irisVis'])}")
        
    # Check overlaps
    # Compare filenames or just check if gallery file exists in raw list (by name or content?)
    # Gallery files are renamed to 'iris_nir.jpg', 'iris_vis.jpg' etc.
    # So we can't match by name.
    # But usually gallery is created by copying.
    
    # Let's inspect the iris_nir.jpg of user_59 and compare it with ALL raw files? 
    # That's too slow.
    # But wait, 2600 files is fast to loop.
    
    print("Comparing Iris NIR...")
    gallery_nir = gallery_path / "iris_nir.jpg"
    if gallery_nir.exists():
        with open(gallery_nir, 'rb') as f:
            g_nir_data = f.read()
            
        matches = 0
        for p in raw_cache.get('irisNir', []):
            try:
                with open(p, 'rb') as f:
                    if f.read() == g_nir_data:
                        print(f"MATCH FOUND: {p}")
                        matches += 1
            except Exception as e:
                pass
        print(f"Total NIR matches in raw: {matches}")
        
    print("Comparing Iris Vis...")
    gallery_vis = gallery_path / "iris_vis.jpg"
    if gallery_vis.exists():
        # Gallery VIS might be converted to JPG from TIFF?
        # If so binary match won't work on raw TIFFs.
        pass

if __name__ == "__main__":
    check_user_59()
