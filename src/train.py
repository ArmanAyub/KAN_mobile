import os
import yaml
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from dataset import get_dataloaders
from model import BaselineClassifier, KANClassifier
from utils import plot_kan_curves
import matplotlib.pyplot as plt
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
    
    inference_times = []
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} {mode}")
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            outputs = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_times.append((time.time() - t0) / images.size(0) * 1000)

            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)[:, 1]
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

    latency = sum(inference_times) / len(inference_times)

    labels_np = np.array(all_labels)
    probs_np  = np.array(all_probs)
    preds_np  = (probs_np >= 0.5).astype(int)

    # True mean AP: average AP across both classes
    ap_pos  = average_precision_score(labels_np, probs_np)
    ap_neg  = average_precision_score(1 - labels_np, 1 - probs_np)
    mean_ap = (ap_pos + ap_neg) / 2.0

    f1        = f1_score(labels_np, preds_np, zero_division=0)
    precision = precision_score(labels_np, preds_np, zero_division=0)
    recall    = recall_score(labels_np, preds_np, zero_division=0)

    return running_loss / total, 100. * correct / total, mean_ap, latency, example_images, f1, precision, recall

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
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({"total_parameters": total_params, "trainable_parameters": trainable_params})
    print(f"Parameters — total: {total_params:,} | trainable: {trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])
    best_acc, best_mAP, best_f1 = 0.0, 0.0, 0.0
    
    for epoch in range(config['train']['epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        do_log = config['logging']['log_images'] and ((epoch % 2 == 0) or (epoch == config['train']['epochs']-1))
        val_loss, val_acc, val_mAP, val_lat, val_imgs, val_f1, val_prec, val_rec = validate(
            model, val_loader, criterion, device, epoch, log_images=do_log)

        print(f"Epoch {epoch+1} | Acc: {val_acc:.2f}% | mAP: {val_mAP:.4f} | F1: {val_f1:.4f} | Latency: {val_lat:.2f}ms")
        wandb.log({"epoch": epoch+1, "val_acc": val_acc, "val_mAP": val_mAP, "val_f1": val_f1,
                   "val_precision": val_prec, "val_recall": val_rec,
                   "val_lat": val_lat, "val_loss": val_loss, "train_acc": train_acc, "train_loss": train_loss})
        if val_imgs: wandb.log({"val_examples": val_imgs})
        
        if val_acc > best_acc:
            best_acc, best_mAP, best_f1 = val_acc, val_mAP, val_f1
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f"models/{model_type}_best.pth")
            wandb.log({"best_val_acc": best_acc, "best_val_mAP": best_mAP, "best_val_f1": best_f1})

    t_acc = best_acc
    if test_loader:
        model.load_state_dict(torch.load(f"models/{model_type}_best.pth", weights_only=True))
        t_loss, t_acc, t_mAP, t_lat, t_imgs, t_f1, t_prec, t_rec = validate(
            model, test_loader, criterion, device, 0, log_images=True, mode="Test")
        print(f"\nFINAL TEST | Acc: {t_acc:.2f}% | mAP: {t_mAP:.4f} | F1: {t_f1:.4f} | Precision: {t_prec:.4f} | Recall: {t_rec:.4f}")
        wandb.log({"test_acc": t_acc, "test_mAP": t_mAP, "test_f1": t_f1,
                   "test_precision": t_prec, "test_recall": t_rec, "test_lat": t_lat})
        
        if model_type == 'kan':
            fig = plot_kan_curves(model)
            if fig:
                wandb.log({"kan_learned_curve": wandb.Image(fig)})
                plt.close(fig)

    os.makedirs('logs', exist_ok=True)
    results = {
        "model": model_type,
        "trainable_parameters": trainable_params,
        "best_val_acc": best_acc,
        "best_val_mAP": best_mAP,
        "best_val_f1": best_f1,
    }
    if test_loader:
        results.update({"test_acc": t_acc, "test_mAP": t_mAP, "test_f1": t_f1,
                        "test_precision": t_prec, "test_recall": t_rec, "test_lat_ms": t_lat})
    with open(f"logs/{model_type}_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)
