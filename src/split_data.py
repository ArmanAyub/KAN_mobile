import os
import shutil
import random
from tqdm import tqdm

def physical_split(source_dir, train_dir, val_dir, split_ratio=0.8):
    """
    Physically splits the training_set into train and val folders.
    """
    classes = ['humans', 'non-humans']
    
    for cls in classes:
        src_cls_path = os.path.join(source_dir, cls)
        train_cls_path = os.path.join(train_dir, cls)
        val_cls_path = os.path.join(val_dir, cls)
        
        os.makedirs(train_cls_path, exist_ok=True)
        os.makedirs(val_cls_path, exist_ok=True)
        
        IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        files = [f for f in os.listdir(src_cls_path) if os.path.splitext(f.lower())[1] in IMAGE_EXTENSIONS]
        random.seed(42)
        random.shuffle(files)
        
        split_idx = int(len(files) * split_ratio)
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        print(f"Moving {cls}: {len(train_files)} to train, {len(val_files)} to val...")
        
        for f in tqdm(train_files, desc=f"Copying {cls} to train"):
            shutil.copy(os.path.join(src_cls_path, f), os.path.join(train_cls_path, f))
            
        for f in tqdm(val_files, desc=f"Copying {cls} to val"):
            shutil.copy(os.path.join(src_cls_path, f), os.path.join(val_cls_path, f))

if __name__ == "__main__":
    physical_split(
        source_dir="data/training_set",
        train_dir="data/train",
        val_dir="data/val"
    )
    print("\nData split complete! You now have data/train and data/val.")
