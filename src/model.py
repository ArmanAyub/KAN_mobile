import torch
import torch.nn as nn
from torchvision import models
try:
    from fastkan import FastKAN
except ImportError:
    print("Warning: fast-kan not installed.")

class BaselineClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(BaselineClassifier, self).__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # FREEZE BACKBONE: Only train the head
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class KANClassifier(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=64):
        super(KANClassifier, self).__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # FREEZE BACKBONE: Only train the head
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # FastKAN head
        self.kan_head = FastKAN([in_features, hidden_dim, num_classes])

    def forward(self, x):
        features = self.backbone(x)
        return self.kan_head(features)
