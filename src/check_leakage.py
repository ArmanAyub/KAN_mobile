import os
import hashlib
from tqdm import tqdm

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def get_image_hashes(directory):
    hashes = {}
    for root, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f.lower())[1] in IMAGE_EXTENSIONS:
                path = os.path.join(root, f)
                with open(path, 'rb') as img_file:
                    img_hash = hashlib.md5(img_file.read()).hexdigest()
                    hashes[img_hash] = path
    return hashes

def check_pair(name_a, hashes_a, name_b, hashes_b):
    overlap = set(hashes_a.keys()) & set(hashes_b.keys())
    status = "WARNING: Leakage detected!" if overlap else "OK: No leakage."
    print(f"  {name_a} vs {name_b}: {len(overlap)} duplicates — {status}")
    if overlap:
        h = next(iter(overlap))
        print(f"    Example: {hashes_a[h]}  <->  {hashes_b[h]}")
    return len(overlap)

if __name__ == "__main__":
    print("Hashing images (this may take a moment)...")
    train_hashes = get_image_hashes("data/train")
    val_hashes   = get_image_hashes("data/val")
    test_hashes  = get_image_hashes("data/test_set")

    print(f"\nImage counts — train: {len(train_hashes)}  val: {len(val_hashes)}  test: {len(test_hashes)}")
    print("\n--- LEAKAGE AUDIT ---")
    total = 0
    total += check_pair("train", train_hashes, "val",  val_hashes)
    total += check_pair("train", train_hashes, "test", test_hashes)
    total += check_pair("val",   val_hashes,   "test", test_hashes)

    print(f"\n{'ALL CLEAR — splits are clean.' if total == 0 else f'TOTAL LEAKS FOUND: {total} — investigate before trusting metrics.'}")
