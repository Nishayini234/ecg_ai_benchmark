# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from model import ECGResNet1D

class AAMIDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, header=None)
        self.X = df.iloc[:, :-1].values.astype(np.float32)
        # 5 AAMI classes: 0: N, 1: S, 2: V, 3: F, 4: Q
        self.y = df.iloc[:, -1].values.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]).unsqueeze(0), torch.tensor(self.y[idx], dtype=torch.long)

def train_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing training on: {device}")

    train_ds = AAMIDataset("data/mitbih_train.csv")
    test_ds = AAMIDataset("data/mitbih_test.csv")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # Compute inverse class frequencies to mitigate extreme class imbalance (N >> S, V, F, Q)
    class_counts = np.bincount(train_ds.y)
    weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = ECGResNet1D(num_classes=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 6
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")

    # Validation Audit
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            preds = torch.argmax(model(signals), dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    aami_names = ["N (Normal)", "S (SVEB)", "V (VEB)", "F (Fusion)", "Q (Unknown)"]
    print("\nAAMI EC57 Benchmark Audit:")
    print(classification_report(all_labels, all_preds, target_names=aami_names, digits=4))

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/ecg_aami_1dcnn.pt")
    print("Model saved to saved_models/ecg_aami_1dcnn.pt")

if __name__ == "__main__":
    train_benchmark()