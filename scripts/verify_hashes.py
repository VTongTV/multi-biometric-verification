
import sys
import os
import hashlib
from pathlib import Path
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "webapp" / "backend"))

from src.api.core.config import DATA_DIR, GALLERY_DIR
# We need to mock functions or import the logic. 
# Since we modified main.py, we can try to import the function if we can avoid running the app.
# But main.py has side effects (creates App).
# Instead, I'll basically replicate the logic here to verify it works on the file system.

GALLERY_HASHES = set()

def get_file_hash(path):
    """Compute MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def test_load_and_check():
    print("Testing Gallery Hashing...")
    
    if not GALLERY_DIR.exists():
        print("Gallery dir broken")
        return

    count = 0
    for user_dir in GALLERY_DIR.iterdir():
        if user_dir.is_dir():
            for file_path in user_dir.glob("*"):
                if file_path.is_file():
                    try:
                        file_hash = get_file_hash(file_path)
                        GALLERY_HASHES.add(file_hash)
                        count += 1
                        if count < 5:
                            print(f"Hashed {file_path.name}: {file_hash}")
                    except Exception as e:
                        print(e)
                        
    print(f"Indexed {count} gallery files. Unique hashes: {len(GALLERY_HASHES)}")
    
    # Now check a known raw file that IS enrolled.
    # user_59 has iris_nir.jpg. Let's find its source in raw.
    # In a real scenario we don't know the mapping backward easily.
    # But we can check if we can find ANY raw file that collides.
    
    raw_dir = DATA_DIR / "raw"
    collisions = 0
    checks = 0
    
    nir_src = raw_dir / "CASIA-Iris-Interval"
    if nir_src.exists():
        for p in nir_src.glob("**/*.jpg"):
            checks += 1
            h = get_file_hash(p)
            if h in GALLERY_HASHES:
                collisions += 1
                if collisions < 5:
                    print(f"Verified Collision: {p} is in Gallery!")
            
            if checks > 500: break
            
    print(f"Checked {checks} raw files, found {collisions} collisions (files that are enrolled).")
    print("Logic verification passed if collisions > 0 (since gallery comes from raw).")

if __name__ == "__main__":
    test_load_and_check()
