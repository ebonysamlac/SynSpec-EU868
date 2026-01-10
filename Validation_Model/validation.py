
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

# CONFIGURATION 
FILE_PATH = "robust10db.pkl" # Ensure this matches the dataset file
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 100 
IMG_SIZE = (64, 64) 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

#PREPROCESSING
def compute_spectrogram(sig, fs=1e6, nperseg=128):
    """Converts complex IQ to a log-spectrogram image."""
    f, t, Sxx = signal.spectrogram(sig, fs, nperseg=nperseg, noverlap=nperseg//2)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-12)
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

def prepare_datasets(task_type):
    print(f"\n--- Loading Data for Task: {task_type.upper()} ---")
    try:
        with open(FILE_PATH, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File {FILE_PATH} not found.")
        return None, None, None
    raw_snapshots = data['snapshots']
    metadata = data['metadata']
    X = []
    y_labels = []
    skipped = 0
    for i, sig in enumerate(raw_snapshots):
        meta = metadata[i]
        if task_type == 'signal':
            label = meta['label']
            y_labels.append(label)
        elif task_type == 'rogue':
            rogue_details = meta.get('rogue_details', {})
            is_rogue = False
            if 'objects' in meta:
                is_rogue = any(o['is_rogue'] for o in meta['objects'])
                has_visible_viol = False
                has_invisible_viol = False
                for o in meta['objects']:
                    if o['is_rogue']:
                        vt = o.get('violation_type')
                        if vt in ['bw', 'oob']: has_visible_viol = True
                        if vt in ['erp', 'dc']: has_invisible_viol = True # dc usually not in obj, but logic holds
                if meta.get('duty_cycle_violation'): has_invisible_viol = True
                # If it is rogue, but has NO visible violations, skip it.
                if is_rogue and not has_visible_viol:
                    skipped += 1
                    continue            
            elif 'rogue_details' in meta:
                is_rogue = meta['regulatory_violations']
                is_bw = rogue_details.get('bw_violation')
                is_oob = rogue_details.get('oob_violation')
                is_erp = rogue_details.get('erp_violation')
                is_dc = meta.get('duty_cycle_violation')
                if is_rogue:
                    if (is_erp or is_dc) and not (is_bw or is_oob):
                        skipped += 1
                        continue
            y_labels.append("Rogue" if is_rogue else "Compliant")
        if hasattr(sig, 'cpu'): sig = sig.cpu().numpy()
        X.append(compute_spectrogram(sig))

    X = np.array(X)[:, np.newaxis, :, :] # (N, 1, 64, 64)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    classes = le.classes_
    print(f"Loaded {len(X)} samples. (Skipped {skipped})")
    print(f"Classes: {classes}")
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    return (train_ds, val_ds, test_ds), classes

# MODEL 
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SimpleClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMG_SIZE[0]//8) * (IMG_SIZE[1]//8), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# TRAINING ######
def run_training_task(task_name, datasets, classes):
    train_ds, val_ds, test_ds = datasets
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    model = SimpleClassifier(len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    print(f"\n>>> Training Task: {task_name} <<<")
    for epoch in range(EPOCHS):
        # TRAIN
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * X.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        t_loss = run_loss / total
        t_acc = 100 * correct / total
        
        # VALIDATION
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                out = model(X)
                loss = criterion(out, y)
                v_loss += loss.item() * X.size(0)
                _, pred = torch.max(out, 1)
                v_correct += (pred == y).sum().item()
                v_total += y.size(0)
        val_loss = v_loss / v_total
        val_acc = 100 * v_correct / v_total
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        if (epoch+1) % 10 == 0:
            print(f"Ep {epoch+1}/{EPOCHS} | T_Loss: {t_loss:.4f} Acc: {t_acc:.2f}% | V_Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    # TEST EVALUATION
    model.eval()
    t_loss, t_correct, t_total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss = criterion(out, y)
            t_loss += loss.item() * X.size(0)
            _, pred = torch.max(out, 1)
            t_correct += (pred == y).sum().item()
            t_total += y.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    test_loss = t_loss / t_total
    test_acc = 100 * t_correct / t_total
    print(f"\n{task_name} FINAL TEST RESULTS:")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix: {task_name}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'cm_{task_name}.png')
    plt.show()
    print(f"\nClassification Report ({task_name}):")
    print(classification_report(all_labels, all_preds, target_names=classes))
    return history

# EXECUTE AND PLOT ######
if __name__ == "__main__":
    data_sig, classes_sig = prepare_datasets('signal')
    if data_sig:
        hist_sig = run_training_task('Signal_Classification', data_sig, classes_sig)
    data_rog, classes_rog = prepare_datasets('rogue')
    if data_rog:
        hist_rog = run_training_task('Regulatory Detection', data_rog, classes_rog)
    if data_sig and data_rog:
        epochs = range(1, EPOCHS + 1)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color='tab:red')
        l1, = ax1.plot(
            epochs, hist_sig['train_loss'],
            color='tab:red', linestyle='-',
            label='Signal Training Loss'
        )
        l2, = ax1.plot(
            epochs, hist_rog['train_loss'],
            color='tab:orange', linestyle='--',
            label='Regulatory Training Loss'
        )
        ax1.set_ylim(0.2, 1.6)
        ax1.set_yticks([i * 0.2 for i in range(1, 9)])
        ax1.tick_params(axis='y', labelcolor='tab:red')    
        ax1.set_xticks(range(0, EPOCHS + 1, 10))
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy (%)', color='tab:blue')
        l3, = ax2.plot(
            epochs, hist_sig['val_acc'],
            color='tab:blue', linestyle='-',
            label='Signal Validation Accuracy'
        )
        l4, = ax2.plot(
            epochs, hist_rog['val_acc'],
            color='tab:cyan', linestyle='--',
            label='Regulatory Validation Accuracy'
        )
        ax2.set_yticks(range(0, 101, 20))
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        lines = [l1, l2, l3, l4]
        ax1.legend(lines, [l.get_label() for l in lines], loc='center right')
        plt.title('Training Dynamics: Signal vs. Regulatory Tasks')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('combined_training_dynamics_10db.png')
        plt.show()
        print("\nSaved combined plot to 'combined_training_dynamics_10db.png'")
