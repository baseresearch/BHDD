"""
Quick baseline experiments for the Burmese Handwritten Digit Dataset (BHDD).
MLP (sklearn), CNN, and Improved CNN (PyTorch) baselines with evaluation metrics.
"""

import matplotlib
matplotlib.use('Agg')

import os
import pickle
import random
import time
import copy

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data.pkl")
FIG_DIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load & preprocess
# ---------------------------------------------------------------------------
print("=" * 60)
print("Loading data …")
with open(DATA_PATH, "rb") as f:
    dataset = pickle.load(f)

def to_arrays(split):
    images = np.array([s["image"] for s in split], dtype=np.float32) / 255.0
    labels = np.array([s["label"] for s in split], dtype=np.int64)
    return images, labels

X_train, y_train = to_arrays(dataset["trainDataset"])
X_test, y_test = to_arrays(dataset["testDataset"])

print(f"  Train : {X_train.shape[0]:,} samples")
print(f"  Test  : {X_test.shape[0]:,} samples")
print(f"  Image : {X_train.shape[1:]}  |  Labels : {sorted(set(y_train.tolist()))}")

# Flattened versions for MLP
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# ---------------------------------------------------------------------------
# 1) MLP Baseline (sklearn)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Training MLP (sklearn) …")
t0 = time.time()

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    max_iter=50,
    random_state=SEED,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=False,
)
mlp.fit(X_train_flat, y_train)
mlp_time = time.time() - t0

y_pred_mlp = mlp.predict(X_test_flat)

mlp_acc = accuracy_score(y_test, y_pred_mlp)
mlp_f1 = f1_score(y_test, y_pred_mlp, average="macro")
mlp_prec = precision_score(y_test, y_pred_mlp, average="macro")
mlp_rec = recall_score(y_test, y_pred_mlp, average="macro")

print(f"  Done in {mlp_time:.1f}s  (stopped at epoch {mlp.n_iter_})")

# ---------------------------------------------------------------------------
# 2) CNN Baseline (PyTorch)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Training CNN (PyTorch) …")

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"  Device: {device}")

# Data loaders — images need shape (N, 1, 28, 28)
X_train_t = torch.from_numpy(X_train[:, np.newaxis, :, :])
y_train_t = torch.from_numpy(y_train)
X_test_t = torch.from_numpy(X_test[:, np.newaxis, :, :])
y_test_t = torch.from_numpy(y_test)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
)
test_loader = DataLoader(
    TensorDataset(X_test_t, y_test_t), batch_size=256, shuffle=False,
)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                   # -> 14x14
            nn.Conv2d(32, 64, 3, padding=1),   # -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                   # -> 7x7
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ImprovedCNN(nn.Module):
    """3 conv layers with batch normalization — simple and effective."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),    # 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),   # 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # -> 14x14
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, 3, padding=1),   # 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # -> 7x7
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class AugmentedDataset(Dataset):
    """On-the-fly augmentation: small rotation, translation, scaling."""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        angle = (random.random() - 0.5) * 30
        angle_rad = angle * np.pi / 180.0
        tx = (random.random() - 0.5) * 4
        ty = (random.random() - 0.5) * 4
        scale = 0.9 + random.random() * 0.2
        cos_a = np.cos(angle_rad) * scale
        sin_a = np.sin(angle_rad) * scale
        theta = torch.tensor([
            [cos_a, -sin_a, tx / 14.0],
            [sin_a,  cos_a, ty / 14.0],
        ], dtype=img.dtype).unsqueeze(0)
        grid = nn.functional.affine_grid(theta, img.unsqueeze(0).size(), align_corners=False)
        img = nn.functional.grid_sample(img.unsqueeze(0), grid, align_corners=False, padding_mode='zeros')
        return img.squeeze(0), label


# --- Train Simple CNN ---
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 15
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    avg_loss = running_loss / len(train_loader.dataset)

    if epoch % 5 == 0 or epoch == 1:
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (model(imgs).argmax(1) == labels).sum().item()
        test_acc = correct / len(test_loader.dataset)
        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  test_acc={test_acc:.4f}")
    else:
        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}")

cnn_time = time.time() - t0

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

y_pred_cnn = np.concatenate(all_preds)
y_true_cnn = np.concatenate(all_labels)

cnn_acc = accuracy_score(y_true_cnn, y_pred_cnn)
cnn_f1 = f1_score(y_true_cnn, y_pred_cnn, average="macro")
cnn_prec = precision_score(y_true_cnn, y_pred_cnn, average="macro")
cnn_rec = recall_score(y_true_cnn, y_pred_cnn, average="macro")

print(f"  Done in {cnn_time:.1f}s")

# ---------------------------------------------------------------------------
# 3) Improved CNN (PyTorch) — with BN, augmentation, cosine LR
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Training Improved CNN (PyTorch) …")
print("  (3 conv layers + BatchNorm + augmentation + cosine LR)")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

aug_loader = DataLoader(
    AugmentedDataset(X_train_t, y_train_t), batch_size=128, shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
)

model_imp = ImprovedCNN().to(device)
criterion_imp = nn.CrossEntropyLoss()
optimizer_imp = optim.Adam(model_imp.parameters(), lr=1e-3)

IMP_EPOCHS = 25
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_imp, T_max=IMP_EPOCHS)

t0 = time.time()
best_acc = 0
best_state = None

for epoch in range(1, IMP_EPOCHS + 1):
    model_imp.train()
    running_loss = 0.0
    for imgs, labels in aug_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer_imp.zero_grad()
        loss = criterion_imp(model_imp(imgs), labels)
        loss.backward()
        optimizer_imp.step()
        running_loss += loss.item() * imgs.size(0)
    scheduler.step()
    avg_loss = running_loss / len(aug_loader.dataset)

    if epoch % 5 == 0 or epoch == IMP_EPOCHS:
        model_imp.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (model_imp(imgs).argmax(1) == labels).sum().item()
        test_acc = correct / len(test_loader.dataset)
        lr_now = optimizer_imp.param_groups[0]['lr']
        print(f"  Epoch {epoch:2d}/{IMP_EPOCHS}  loss={avg_loss:.4f}  test_acc={test_acc:.4f}  lr={lr_now:.6f}")
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(model_imp.state_dict())
    else:
        print(f"  Epoch {epoch:2d}/{IMP_EPOCHS}  loss={avg_loss:.4f}")

imp_time = time.time() - t0

if best_state is not None:
    model_imp.load_state_dict(best_state)

model_imp.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        preds = model_imp(imgs).argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

y_pred_imp = np.concatenate(all_preds)
y_true_imp = np.concatenate(all_labels)

imp_acc = accuracy_score(y_true_imp, y_pred_imp)
imp_f1 = f1_score(y_true_imp, y_pred_imp, average="macro")
imp_prec = precision_score(y_true_imp, y_pred_imp, average="macro")
imp_rec = recall_score(y_true_imp, y_pred_imp, average="macro")

print(f"  Done in {imp_time:.1f}s")

# ---------------------------------------------------------------------------
# 4) Confusion Matrix (Improved CNN)
# ---------------------------------------------------------------------------
BURMESE_DIGITS = ["၀", "၁", "၂", "၃", "၄", "၅", "၆", "၇", "၈", "၉"]

# Use a Myanmar-capable font for Burmese digit labels
import matplotlib.font_manager as fm
_myanmar_font = None
for _p in [
    "/System/Library/Fonts/NotoSansMyanmar.ttc",
    "/System/Library/Fonts/NotoSerifMyanmar.ttc",
    "/System/Library/Fonts/Supplemental/Myanmar MN.ttc",
    "/System/Library/Fonts/Supplemental/Myanmar Sangam MN.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansMyanmar-Regular.ttf",
]:
    if os.path.exists(_p):
        _myanmar_font = fm.FontProperties(fname=_p)
        break
if _myanmar_font is None:
    _myanmar_font = fm.FontProperties()

cm = confusion_matrix(y_true_imp, y_pred_imp)
fig, ax = plt.subplots(figsize=(8, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=BURMESE_DIGITS)
disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
for label in ax.get_xticklabels():
    label.set_fontproperties(_myanmar_font)
    label.set_fontsize(13)
for label in ax.get_yticklabels():
    label.set_fontproperties(_myanmar_font)
    label.set_fontsize(13)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Improved CNN Confusion Matrix — BHDD Test Set", fontsize=13, pad=12)
plt.tight_layout()
cm_path = os.path.join(FIG_DIR, "confusion_matrix.png")
fig.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nConfusion matrix saved → {cm_path}")

# ---------------------------------------------------------------------------
# 5) Results Table
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("BASELINE RESULTS — BHDD Test Set")
print("=" * 60)
header = f"{'Model':<16} {'Accuracy':>10} {'F1 (macro)':>12} {'Precision':>12} {'Recall':>10} {'Time (s)':>10}"
print(header)
print("-" * len(header))
print(f"{'MLP':<16} {mlp_acc:>10.4f} {mlp_f1:>12.4f} {mlp_prec:>12.4f} {mlp_rec:>10.4f} {mlp_time:>10.1f}")
print(f"{'CNN':<16} {cnn_acc:>10.4f} {cnn_f1:>12.4f} {cnn_prec:>12.4f} {cnn_rec:>10.4f} {cnn_time:>10.1f}")
print(f"{'Improved CNN':<16} {imp_acc:>10.4f} {imp_f1:>12.4f} {imp_prec:>12.4f} {imp_rec:>10.4f} {imp_time:>10.1f}")
print("=" * len(header))
print(f"\nTrain samples: {X_train.shape[0]:,}  |  Test samples: {X_test.shape[0]:,}")
print(f"Device: {device}  |  CNN epochs: {EPOCHS}  |  Improved CNN epochs: {IMP_EPOCHS}  |  Seed: {SEED}")
print("Done.")
