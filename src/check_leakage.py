import os
import hashlib
from tqdm import tqdm

def get_image_hashes(directory):
    hashes = {}
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith('.jpg'):
                path = os.path.join(root, f)
                with open(path, 'rb') as img_file:
                    img_hash = hashlib.md5(img_file.read()).hexdigest()
                    hashes[img_hash] = path
    return hashes

def check_for_leakage(train_dir, val_dir):
    print("Calculating hashes for Train set...")
    train_hashes = get_image_hashes(train_dir)
    
    print("Calculating hashes for Val set...")
    val_hashes = get_image_hashes(val_dir)
    
    train_keys = set(train_hashes.keys())
    val_keys = set(val_hashes.keys())
    
    intersection = train_keys.intersection(val_keys)
    
    print("\n--- LEAKAGE AUDIT RESULTS ---")
    print(f"Total Unique Train Images: {len(train_keys)}")
    print(f"Total Unique Val Images:   {len(val_keys)}")
    print(f"Identical Images Found in Both: {len(intersection)}")
    
    if len(intersection) > 0:
        print("\nWARNING: Data Leakage detected!")
        print("Example of duplicate:")
        dup_hash = list(intersection)[0]
        print(f"Train path: {train_hashes[dup_hash]}")
        print(f"Val path:   {val_hashes[dup_hash]}")
    else:
        print("\nSUCCESS: No data leakage found. Your mAP of 1.0 is likely genuine.")

if __name__ == "__main__":
    check_for_leakage("data/train", "data/val")
