import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy import signal

# --- CONFIGURATION ---
FILE_PATH = "robust10db.pkl"
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 100
IMG_SIZE = (64, 64) # 

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# --- 1. DATA PREPROCESSING ---
def compute_spectrogram(sig, fs=1e6, nperseg=128):
    #Converts complex IQ to a log-spectrogram image
    f, t, Sxx = signal.spectrogram(sig, fs, nperseg=nperseg, noverlap=nperseg//2)
    Sxx = np.fft.fftshift(Sxx, axes=0)

    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-12)
    
    # Normalize 
    Sxx_norm = (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min())
    
    target_h, target_w = IMG_SIZE
    
    curr_h, curr_w = Sxx_norm.shape
    
    start_h = (curr_h - target_h) // 2
    start_w = (curr_w - target_w) // 2
    
    if curr_h < target_h or curr_w < target_w:
        padded = np.zeros(IMG_SIZE)
        h = min(curr_h, target_h)
        w = min(curr_w, target_w)
        padded[:h, :w] = Sxx_norm[:h, :w]
        return padded
        
    return Sxx_norm[start_h:start_h+target_h, start_w:start_w+target_w]

def load_and_process_data():
    print("Loading dataset...")
    with open(FILE_PATH, 'rb') as f:
        data = pickle.load(f)
        
    raw_snapshots = data['snapshots']
    metadata = data['metadata']
    X = []
    y_labels = []
    
    print(f"Processing {len(raw_snapshots)} samples into spectrograms...")
    for i, sig in enumerate(raw_snapshots):
        if hasattr(sig, 'cpu'): sig = sig.cpu().numpy()        
        spec = compute_spectrogram(sig)
        X.append(spec)
        label = metadata[i]['label']
        y_labels.append(label)

    X = np.array(X) # Shape: (N, 64, 64)
    X = X[:, np.newaxis, :, :] # Add Channel Dim: (N, 1, 64, 64)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    classes = le.classes_
    
    return X, y_encoded, classes

# DATA SPLITTING (70/15/15)
def create_splits(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"\nData Splits:")
    print(f"Train: {len(X_train)} samples (70%)")
    print(f"Val:   {len(X_val)} samples (15%)")
    print(f"Test:  {len(X_test)} samples (15%)")
  
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    return train_ds, val_ds, test_ds

# -- MODEL ---
class SignalClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SignalClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# TRAINING & EVALUATION 
def train_model():
    X, y, class_names = load_and_process_data()
    train_ds, val_ds, test_ds = create_splits(X, y)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    model = SignalClassifier(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n--- Starting Training ---")
    train_losses, val_accuracies = [], []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

    # TESTING & CONFUSION MATRIX 
    print("\n--- Final Test Evaluation ---")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrixTest10db.png')
    print("Saved 'confusion_matrixTest10db.png'")

    print("\nConfusion Matrix (Text Table):")    
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Error: {e}")
