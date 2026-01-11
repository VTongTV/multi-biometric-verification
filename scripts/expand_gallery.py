import os
import shutil
from pathlib import Path
import argparse

def expand_gallery(target_count=100):
    # Setup paths
    base_dir = Path("webapp/backend/src/app/static/gallery")
    
    if not base_dir.exists():
        print(f"Error: Gallery directory not found at {base_dir}")
        return

    existing_users = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    current_count = len(existing_users)
    
    if current_count == 0:
        print("Error: No existing users to clone.")
        return

    print(f"Current gallery size: {current_count}")
    
    if current_count >= target_count:
        print(f"Gallery already has {current_count} users (>= {target_count}). No action needed.")
        return

    # Use user_01 as the template
    template_user = base_dir / "user_01"
    if not template_user.exists():
        # Fallback to first available
        template_user = existing_users[0]
    
    print(f"Cloning from template: {template_user.name}")

    for i in range(current_count + 1, target_count + 1):
        new_user_name = f"user_{i:02d}"
        new_user_path = base_dir / new_user_name
        
        if new_user_path.exists():
            continue
            
        try:
            shutil.copytree(template_user, new_user_path)
            # print(f"Created {new_user_name}") 
        except Exception as e:
            print(f"Failed to create {new_user_name}: {e}")

    print(f"Expansion complete. Gallery now has {target_count} users.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand gallery with synthetic users")
    parser.add_argument("--count", type=int, default=100, help="Target total number of users")
    args = parser.parse_args()
    
    expand_gallery(args.count)
