import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import scipy.io as sio
from PIL import Image
import numpy as np
import os
import glob
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch.nn.functional as F


# [0] Seed & directory setup
torch.manual_seed(42)
torch.cuda.manual_seed(42)
os.makedirs('saved_models', exist_ok=True)

# ========== 1. Paths & Hyperparameters ==========
DATA_DIR  = r'./data/images'
LABEL_DIR = r'./data/labels'

MODEL_NAME        = "SegFormer"
SEGFORMER_VARIANT = "B2"

K_FOLDS          = 5
EPOCHS_PER_FOLD  = 200
BATCH_SIZE       = 2
LR               = 0.0001
CLASSES_NUM      = 4
CLASS_WEIGHTS    = torch.tensor([0.4, 1.5, 2.0, 1.0])

EARLY_STOP_PATIENCE = 50  # Stop training if no improvement for 50 epochs

# ========== 2. Data Loading ==========
def load_data_and_stratify(d_path, l_path):
    d_files = sorted(glob.glob(os.path.join(d_path, '*.mat')))
    l_files = sorted(glob.glob(os.path.join(l_path, '*.png')))
    data_list, label_list, density_ratios = [], [], []

    print("Loading data and analyzing density...")
    for df, lf in zip(d_files, l_files):
        data  = sio.loadmat(df)['image']
        label = np.array(Image.open(lf))

        ratio = np.mean(label != 0)
        density_ratios.append(ratio)
        data_list.append(data)
        label_list.append(label)

    density_ratios = np.array(density_ratios)
    sort_idx = np.argsort(density_ratios)
    strat_tags = np.zeros(len(density_ratios), dtype=int)

    num_bins = 5
    for i in range(num_bins):
        start = i * len(density_ratios) // num_bins
        end   = (i + 1) * len(density_ratios) // num_bins
        strat_tags[sort_idx[start:end]] = i

    return (
        torch.tensor(np.array(data_list)).float(),
        torch.tensor(np.array(label_list)).long(),
        strat_tags
    )

# ========== 3. Augmentation ==========
def apply_augment(data, labels):
    # Horizontal & Vertical Flip
    if torch.rand(1) < 0.5:
        data   = torch.flip(data,   [-1])
        labels = torch.flip(labels, [-1])
    if torch.rand(1) < 0.5:
        data   = torch.flip(data,   [-2])
        labels = torch.flip(labels, [-2])

    # Random 90-degree rotation (no interpolation distortion)
    if torch.rand(1) < 0.5:
        k      = torch.randint(1, 4, (1,)).item()
        data   = torch.rot90(data,   k, dims=[-2, -1])
        labels = torch.rot90(labels, k, dims=[-2, -1])

    return data, labels

# ========== 4. Model Builder ==========
def get_model(nc, n_class):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=n_class,
        ignore_mismatched_sizes=True
    )

    # Replace input projection layer to accept nc-channel hyperspectral input
    # Acts as a learnable dimensionality reduction (similar to PCA)
    embed_dim = model.segformer.encoder.patch_embeddings[0].proj.out_channels
    model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
        nc, embed_dim, kernel_size=7, stride=4, padding=3
    )

    return model

# ========== 5. Evaluation Metric ==========
def get_non_bg_acc(preds, labels):
    """Accuracy excluding background class (label != 0)"""
    mask = (labels != 0)
    if mask.sum() == 0:
        return 0.0
    return (preds[mask] == labels[mask]).float().mean().item() * 100

# ========== 6. Main Training Loop ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
full_data, full_labels, strat_tags = load_data_and_stratify(DATA_DIR, LABEL_DIR)
NC = full_data.shape[3]

skf         = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
cv_scores   = []
best_epochs = []

for fold, (t_idx, v_idx) in enumerate(skf.split(full_data, strat_tags)):
    model     = get_model(NC, CLASSES_NUM).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))

    t_loader = DataLoader(TensorDataset(full_data[t_idx], full_labels[t_idx]),
                          batch_size=BATCH_SIZE, shuffle=True)
    v_loader = DataLoader(TensorDataset(full_data[v_idx], full_labels[v_idx]),
                          batch_size=BATCH_SIZE, shuffle=False)

    best_fold_acc      = 0
    best_epoch         = 0
    early_stop_counter = 0

    print(f"\nFold {fold+1} training started")

    for epoch in range(EPOCHS_PER_FOLD):
        # Training
        model.train()
        train_loss = 0
        for data, target in t_loader:
            data   = data.permute(0, 3, 1, 2).to(device)
            target = target.to(device)
            data, target = apply_augment(data, target)

            optimizer.zero_grad()
            logits = model(data).logits

            if logits.shape[-2:] != target.shape[-2:]:
                logits = nn.functional.interpolate(logits, size=target.shape[1:], mode='bilinear')

            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for vd, vl in v_loader:
                vd       = vd.permute(0, 3, 1, 2).to(device)
                vl       = vl.to(device)
                v_logits = model(vd).logits
                if v_logits.shape[-2:] != vl.shape[-2:]:
                    v_logits = nn.functional.interpolate(v_logits, size=vl.shape[1:], mode='bilinear')
                all_p.append(torch.argmax(v_logits, dim=1).cpu())
                all_l.append(vl.cpu())

        valid_acc = get_non_bg_acc(torch.cat(all_p), torch.cat(all_l))

        # Learning rate scheduler step
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(valid_acc)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < prev_lr:
            print(f"    [LR Reduced] {prev_lr:.6f} -> {current_lr:.6f}")

        # Early stopping & model checkpoint
        if valid_acc > best_fold_acc and epoch >= 15:
            best_fold_acc      = valid_acc
            best_epoch         = epoch + 1
            early_stop_counter = 0
            torch.save(model.state_dict(), f'saved_models/best_model_fold_{fold+1}.pth')
        else:
            early_stop_counter += 1

        if (epoch + 1) % 5 == 0 or early_stop_counter == 0:
            print(f"Epoch {epoch+1:>3} | Loss: {train_loss/len(t_loader):.4f} | "
                  f"Acc: {valid_acc:.2f}% | Best: {best_fold_acc:.2f}% | "
                  f"Counter: {early_stop_counter}/{EARLY_STOP_PATIENCE}")

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"[Early Stop] Training stopped at epoch {epoch+1} (no improvement)")
            break

    print(f"Fold {fold+1} done | Best: {best_fold_acc:.2f}% (epoch {best_epoch})")
    cv_scores.append(best_fold_acc)
    best_epochs.append(best_epoch)

# ========== 7. Final Summary ==========
avg_best_epoch = int(np.round(np.mean(best_epochs)))

print("\n" + "=" * 50)
print(f"Mean Accuracy (excl. background): {np.mean(cv_scores):.2f}% (+/-{np.std(cv_scores):.2f})")
print(f"Best epoch per fold: {best_epochs}")
print(f"Recommended full-data training epochs: {avg_best_epoch}")
print("=" * 50)