import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32, img_size=224):
    """
    Loads data from fixed train, val, and test folders.
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test_set')
    
    # 1. Training Transform (Augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 2. Validation & Test Transform (Static)
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 3. Load Datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
    
    test_dataset = None
    if os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)
        print(f"Loaded test set with {len(test_dataset)} images.")
    
    # 4. Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset.classes

if __name__ == "__main__":
    try:
        train_loader, val_loader, test_loader, classes = get_dataloaders("data")
        print(f"Loaded successfully from fixed folders. Classes: {classes}")
    except Exception as e:
        print(f"Ensure you run 'python src/split_data.py' first: {e}")
