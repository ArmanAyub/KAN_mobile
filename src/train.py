import os
import yaml
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from dataset import get_dataloaders
from model import BaselineClassifier, KANClassifier
from utils import plot_kan_curves
import wandb

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix(loss=running_loss/total, acc=100. * correct/total)
    return running_loss / total, 100. * correct / total

def validate(model, loader, criterion, device, epoch, log_images=False, mode="Validation"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs, example_images = [], [], []
    
    start_time = time.time()
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} {mode}")
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)[:, 1] # Probability for 'Non-Human' (class 1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if log_images and i == 0:
                for j in range(min(len(images), 8)):
                    img = images[j].cpu()
                    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    example_images.append(wandb.Image(img, caption=f"Pred: {predicted[j].item()}, Actual: {labels[j].item()}"))
    
    latency = (time.time() - start_time) / total * 1000 # ms per image
    mAP = average_precision_score(all_labels, all_probs)
    return running_loss / total, 100. * correct / total, mAP, latency, example_images

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = config['model']['type']
    
    wandb.init(project=config['logging']['project_name'], name=f"{model_type}-run", config=config)
    
    # Updated to new simplified get_dataloaders
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=config['dataset']['data_dir'], 
        batch_size=config['dataset']['batch_size'],
        img_size=config['dataset']['image_size']
    )
        
    if model_type == 'baseline':
        model = BaselineClassifier(num_classes=len(classes))
    else:
        model = KANClassifier(num_classes=len(classes), hidden_dim=config['model']['hidden_dim'])
        
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    wandb.config.update({"total_parameters": total_params})
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])
    best_acc, best_mAP = 0.0, 0.0
    
    for epoch in range(config['train']['epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        do_log = config['logging']['log_images'] and ((epoch % 2 == 0) or (epoch == config['train']['epochs']-1))
        val_loss, val_acc, val_mAP, val_lat, val_imgs = validate(model, val_loader, criterion, device, epoch, log_images=do_log)
        
        print(f"Epoch {epoch+1} | Acc: {val_acc:.2f}% | mAP: {val_mAP:.4f}")
        wandb.log({"epoch": epoch+1, "val_acc": val_acc, "val_mAP": val_mAP, "val_lat": val_lat, "val_loss": val_loss, "train_acc": train_acc, "train_loss": train_loss})
        if val_imgs: wandb.log({"val_examples": val_imgs})
        
        if val_acc > best_acc:
            best_acc, best_mAP = val_acc, val_mAP
            torch.save(model.state_dict(), f"models/{model_type}_best.pth")

    if test_loader:
        model.load_state_dict(torch.load(f"models/{model_type}_best.pth"))
        t_loss, t_acc, t_mAP, t_lat, t_imgs = validate(model, test_loader, criterion, device, 0, log_images=True, mode="Test")
        print(f"\nFINAL TEST | Acc: {t_acc:.2f}% | mAP: {t_mAP:.4f}")
        wandb.log({"test_acc": t_acc, "test_mAP": t_mAP, "test_lat": t_lat})
        
        if model_type == 'kan':
            fig = plot_kan_curves(model)
            if fig:
                wandb.log({"kan_learned_curve": wandb.Image(fig)})
                plt.close(fig)

    os.makedirs('logs', exist_ok=True)
    with open(f"logs/{model_type}_results.json", 'w') as f:
        json.dump({"model": model_type, "test_acc": t_acc if test_loader else best_acc}, f, indent=4)
    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)
