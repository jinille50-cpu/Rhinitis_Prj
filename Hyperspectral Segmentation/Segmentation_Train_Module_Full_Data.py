import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
from PIL import Image
import numpy as np
import os
import glob
from transformers import SegformerForSemanticSegmentation

torch.manual_seed(42)
torch.cuda.manual_seed(42)
os.makedirs('saved_models', exist_ok=True)

# ========== 1. Paths & Hyperparameters ==========
DATA_DIR  = r'./data/images'
LABEL_DIR = r'./data/labels'

CLASSES_NUM   = 4
CLASS_WEIGHTS = torch.tensor([0.4, 1.5, 2.0, 1.0])
BATCH_SIZE    = 2
LR            = 0.0001

# Set this from the K-fold result (avg_best_epoch)
FULL_TRAIN_EPOCHS = 120  # <-- Replace with avg_best_epoch from cross-validation

# ========== 2. Data Loading ==========
def load_all_data(d_path, l_path):
    d_files = sorted(glob.glob(os.path.join(d_path, '*.mat')))
    l_files = sorted(glob.glob(os.path.join(l_path, '*.png')))
    data_list, label_list = [], []

    print("Loading full dataset...")
    for df, lf in zip(d_files, l_files):
        data  = sio.loadmat(df)['image']
        label = np.array(Image.open(lf))
        label[label == 3] = 2  # Merge label 3 into label 2
        data_list.append(data)
        label_list.append(label)

    print(f"Total samples: {len(data_list)}")
    return (
        torch.tensor(np.array(data_list)).float(),
        torch.tensor(np.array(label_list)).long()
    )

# ========== 3. Augmentation ==========
def apply_augment(data, labels):
    if torch.rand(1) < 0.5:
        data   = torch.flip(data,   [-1])
        labels = torch.flip(labels, [-1])
    if torch.rand(1) < 0.5:
        data   = torch.flip(data,   [-2])
        labels = torch.flip(labels, [-2])
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
    embed_dim = model.segformer.encoder.patch_embeddings[0].proj.out_channels
    model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
        nc, embed_dim, kernel_size=7, stride=4, padding=3
    )
    return model

# ========== 5. Main Full Training ==========
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
full_data, full_labels = load_all_data(DATA_DIR, LABEL_DIR)
NC = full_data.shape[3]

loader    = DataLoader(TensorDataset(full_data, full_labels),
                       batch_size=BATCH_SIZE, shuffle=True)
model     = get_model(NC, CLASSES_NUM).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))

print(f"Training on full data for {FULL_TRAIN_EPOCHS} epochs...")

for epoch in range(FULL_TRAIN_EPOCHS):
    model.train()
    train_loss = 0

    for data, target in loader:
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

    avg_loss = train_loss / len(loader)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:>3}/{FULL_TRAIN_EPOCHS} | Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), 'saved_models/final_production_model.pth')
print("Full training complete. Model saved to 'saved_models/final_production_model.pth'")