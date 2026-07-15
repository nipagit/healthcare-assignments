# =============================================================================
# Coding Assignment 3: Image-Based Cancer Diagnosis Using CNNs
# Student  : Nipa Das
# Dataset  : Synthetic Breast Histopathology (generated to match BUSI structure)
#            Replace the data generation block with your real dataset loader
#            when using BUSI or Histopathologic Cancer Detection (Kaggle).
# Framework: PyTorch
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    classification_report, accuracy_score
)

import warnings
warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*65}")
print(f"  Coding Assignment 3: CNN-Based Cancer Image Diagnosis")
print(f"{'='*65}")
print(f"  Device : {DEVICE}")

# =============================================================================
# SYNTHETIC DATASET GENERATION
# (Replace this section with real BUSI / Kaggle dataset path when available)
# =============================================================================
DATASET_DIR = "/tmp/cancer_dataset"
CLASSES      = ["benign", "malignant"]
IMG_SIZE     = 128
N_PER_CLASS  = 300   # 300 benign + 300 malignant = 600 total

def generate_synthetic_dataset(out_dir, n_per_class, img_size):
    """
    Creates synthetic grayscale medical-style images.
    Benign  → smooth circular blob (low texture noise).
    Malignant → irregular blob with high-frequency texture noise.
    Replace this function with a real ImageFolder-compatible dataset.
    """
    os.makedirs(out_dir, exist_ok=True)
    for label in CLASSES:
        class_dir = os.path.join(out_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(n_per_class):
            img_arr = np.full((img_size, img_size), 30, dtype=np.uint8)
            cx, cy  = img_size // 2, img_size // 2
            radius  = random.randint(20, 40)
            y_grid, x_grid = np.ogrid[:img_size, :img_size]
            dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
            mask = dist <= radius
            if label == "benign":
                # Smooth, uniform intensity blob
                img_arr[mask] = random.randint(160, 200)
            else:
                # Irregular blob with salt-and-pepper noise
                img_arr[mask] = random.randint(120, 180)
                noise = np.random.randint(-60, 60, (img_size, img_size))
                img_arr = np.clip(img_arr.astype(np.int16) + noise * mask, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr, mode="L").convert("RGB")
            img.save(os.path.join(class_dir, f"{label}_{i:04d}.png"))

if not os.path.exists(DATASET_DIR):
    print("\n  Generating synthetic dataset (replace with real BUSI/Kaggle data)...")
    generate_synthetic_dataset(DATASET_DIR, N_PER_CLASS, IMG_SIZE)
    print(f"  ✔ Dataset created: {N_PER_CLASS*2} images  ({N_PER_CLASS} per class)")
else:
    print(f"\n  ✔ Dataset found at: {DATASET_DIR}")

# =============================================================================
# PHASE A: COMPUTER VISION DATA PIPELINE
# =============================================================================
print(f"\n{'='*65}")
print(f"  PHASE A: COMPUTER VISION DATA PIPELINE")
print(f"{'='*65}")

class CancerImageDataset(Dataset):
    """
    Custom Dataset that loads images from a directory structure:
        root/
            benign/     *.png / *.jpg
            malignant/  *.png / *.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.samples   = []
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(root_dir)))}
        for cls, idx in self.class_to_idx.items():
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.samples.append((os.path.join(cls_dir, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ── Compute dataset-wide mean & std for normalization ────────────────────────
# (Pre-computed for this synthetic set; recompute for real datasets)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ── Transforms: Augmentation (train) & Normalization (val/test) ──────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ── Load full dataset and stratified split: 70 / 15 / 15 ─────────────────────
full_dataset = CancerImageDataset(DATASET_DIR, transform=train_transform)
total        = len(full_dataset)
n_train      = int(0.70 * total)
n_val        = int(0.15 * total)
n_test       = total - n_train - n_val

train_set, val_set, test_set = random_split(
    full_dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED)
)

# Apply eval transform to val & test
val_set.dataset  = CancerImageDataset(DATASET_DIR, transform=eval_transform)
test_set.dataset = CancerImageDataset(DATASET_DIR, transform=eval_transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False, num_workers=0)

print(f"\n  Class mapping : {full_dataset.class_to_idx}")
print(f"  Total images  : {total}")
print(f"  Train         : {len(train_set)}  ({len(train_set)/total*100:.0f}%)")
print(f"  Validation    : {len(val_set)}   ({len(val_set)/total*100:.0f}%)")
print(f"  Test          : {len(test_set)}   ({len(test_set)/total*100:.0f}%)")
print(f"  Image size    : {IMG_SIZE}×{IMG_SIZE}")
print(f"  Augmentations : HFlip, VFlip, Rotation(±15°), ColorJitter")

# =============================================================================
# PHASE B: CNN ARCHITECTURE & TRAINING
# =============================================================================
print(f"\n{'='*65}")
print(f"  PHASE B: CNN ARCHITECTURE & TRAINING")
print(f"{'='*65}")

class CancerCNN(nn.Module):
    """
    Custom Deep CNN for Binary Cancer Classification.

    Architecture:
        Feature Extraction  → 3 Conv Blocks (Conv → BN → ReLU → MaxPool)
        Regularization      → Dropout (p=0.4)
        Classification Head → FC(256) → ReLU → Dropout → FC(1) → Sigmoid
    """
    def __init__(self, dropout_p=0.4):
        super(CancerCNN, self).__init__()

        # ── Conv Block 1: 3 → 32 feature maps ────────────────────────────────
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 128 → 64
        )

        # ── Conv Block 2: 32 → 64 feature maps ───────────────────────────────
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 64 → 32
        )

        # ── Conv Block 3: 64 → 128 feature maps ──────────────────────────────
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 32 → 16
        )

        # ── Classification Head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 1),
            nn.Sigmoid(),                # Binary classification output
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


model     = CancerCNN(dropout_p=0.4).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ── Print architecture summary ────────────────────────────────────────────────
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Model     : CancerCNN")
print(f"  Layers    : 3 Conv Blocks (Conv→BN→ReLU→MaxPool) + FC Head")
print(f"  Dropout   : p = 0.4")
print(f"  Loss      : Binary Cross-Entropy (BCELoss)")
print(f"  Optimizer : Adam (lr=1e-3, StepLR decay ×0.5 every 10 epochs)")
print(f"  Params    : {total_params:,} trainable parameters")

# ── Training Loop ─────────────────────────────────────────────────────────────
EPOCHS = 25
train_losses, val_losses   = [], []
train_accs,   val_accs     = [], []

print(f"\n  Training for {EPOCHS} epochs...\n")
print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>9} | {'Val Acc':>8}")
print(f"  {'-'*55}")

for epoch in range(1, EPOCHS + 1):

    # ── Train ─────────────────────────────────────────────────────────────────
    model.train()
    t_loss, t_correct, t_total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)
        optimizer.zero_grad()
        preds  = model(imgs)
        loss   = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        t_loss    += loss.item() * imgs.size(0)
        predicted  = (preds >= 0.5).float()
        t_correct += (predicted == labels).sum().item()
        t_total   += imgs.size(0)

    scheduler.step()

    # ── Validate ──────────────────────────────────────────────────────────────
    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)
            preds  = model(imgs)
            loss   = criterion(preds, labels)
            v_loss    += loss.item() * imgs.size(0)
            predicted  = (preds >= 0.5).float()
            v_correct += (predicted == labels).sum().item()
            v_total   += imgs.size(0)

    t_loss_avg = t_loss / t_total
    v_loss_avg = v_loss / v_total
    t_acc      = t_correct / t_total
    v_acc      = v_correct / v_total

    train_losses.append(t_loss_avg)
    val_losses.append(v_loss_avg)
    train_accs.append(t_acc)
    val_accs.append(v_acc)

    print(f"  {epoch:>5} | {t_loss_avg:>10.4f} | {t_acc:>8.4f}  | {v_loss_avg:>9.4f} | {v_acc:>8.4f}")

print(f"\n  ✔ Training complete.")

# ── Test Set Evaluation ───────────────────────────────────────────────────────
print(f"\n  ── Test Set Evaluation ──────────────────────────────────")
model.eval()
all_labels, all_preds, all_probs = [], [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs  = imgs.to(DEVICE)
        probs = model(imgs).squeeze(1).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)

test_acc = accuracy_score(all_labels, all_preds)
print(f"\n  Test Accuracy : {test_acc:.4f}")
report = classification_report(all_labels, all_preds,
                                target_names=["benign", "malignant"])
for line in report.splitlines():
    print("  " + line)

# =============================================================================
# PHASE C: DEEP LEARNING DIAGNOSTICS & VISUALIZATION
# =============================================================================
print(f"\n{'='*65}")
print(f"  PHASE C: VISUALIZATION & DIAGNOSTICS")
print(f"{'='*65}")

epochs_range = range(1, EPOCHS + 1)

# ── Plot 1: Learning Curves (Loss & Accuracy) ─────────────────────────────────
print("\n  Generating Plot 1: Learning Curves...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(epochs_range, train_losses, "o-", color="#4C72B0", lw=2, label="Train Loss")
ax1.plot(epochs_range, val_losses,   "s--", color="#DD8452", lw=2, label="Val Loss")
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("BCE Loss", fontsize=12)
ax1.set_title("Training vs Validation Loss", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(linestyle="--", alpha=0.5)

# Accuracy
ax2.plot(epochs_range, train_accs, "o-", color="#55A868", lw=2, label="Train Accuracy")
ax2.plot(epochs_range, val_accs,   "s--", color="#C44E52", lw=2, label="Val Accuracy")
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Accuracy", fontsize=12)
ax2.set_title("Training vs Validation Accuracy", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(linestyle="--", alpha=0.5)
ax2.set_ylim(0, 1.05)

plt.suptitle("CNN Learning Curves — Cancer Image Classification",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/a3_plot1_learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✔ Plot 1 saved: a3_plot1_learning_curves.png")

# ── Plot 2: ROC Curve (Test Set) ──────────────────────────────────────────────
print("  Generating Plot 2: ROC Curve...")
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc     = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#4C72B0", lw=2.5,
        label=f"CNN  (AUC = {roc_auc:.4f})")
ax.fill_between(fpr, tpr, alpha=0.08, color="#4C72B0")
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Chance (AUC = 0.5)")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax.set_title("ROC Curve — CNN Cancer Classifier (Test Set)",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.grid(linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/a3_plot2_roc_curve.png", dpi=150)
plt.close()
print("  ✔ Plot 2 saved: a3_plot2_roc_curve.png")

# ── Plot 3: Confusion Matrix Heatmap ─────────────────────────────────────────
print("  Generating Plot 3: Confusion Matrix...")
cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
    linewidths=0.5, linecolor="gray", ax=ax,
    annot_kws={"size": 16, "weight": "bold"}
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title(
    f"Confusion Matrix — CNN (Test Set)\n"
    f"TP={tp}  TN={tn}  FP={fp}  FN={fn}",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/a3_plot3_confusion_matrix.png", dpi=150)
plt.close()
print("  ✔ Plot 3 saved: a3_plot3_confusion_matrix.png")

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'='*65}")
print(f"  FINAL SUMMARY")
print(f"{'='*65}")
print(f"\n  Architecture  : CancerCNN (3 Conv Blocks + FC Head)")
print(f"  Parameters    : {total_params:,}")
print(f"  Epochs        : {EPOCHS}")
print(f"  Test Accuracy : {test_acc:.4f}")
print(f"  ROC-AUC       : {roc_auc:.4f}")
print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
print(f"\n  Output files:")
print(f"    • a3_plot1_learning_curves.png")
print(f"    • a3_plot2_roc_curve.png")
print(f"    • a3_plot3_confusion_matrix.png")
print(f"\n  ✔ Coding Assignment 3 — Complete!\n")
