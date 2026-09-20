# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGResNet1D(nn.Module):
    """
    1D-CNN architecture designed for AAMI 5-class heartbeat classification.
    Includes Monte Carlo Dropout for uncertainty quantification and accessible
    feature maps for 1D Grad-CAM calculation.
    """
    def __init__(self, num_classes=5):
        super(ECGResNet1D, self).__init__()
        
        # Block 1: Low-level morphological feature extraction
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        # Block 2: Intermediate wave deflections
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        # Block 3: Target layer for 1D Grad-CAM
        self.target_conv = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier with Dropout for Monte Carlo Uncertainty Estimation
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Activations for Grad-CAM
        x = F.relu(self.bn3(self.target_conv(x)))
        
        x_pooled = self.global_pool(x).squeeze(-1)
        x_drop = self.dropout(x_pooled)
        logits = self.fc(x_drop)
        return logits