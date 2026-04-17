import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
from PIL import Image
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
from transformers import SegformerForSemanticSegmentation
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# [0] Configuration & output directory
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES_NUM = 3  # 0: Background, 1: Eosinophil, 2: Edema
CLASS_NAMES = ['Background', 'Eosinophil', 'Edema']

SAVE_DIR = "paper_figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== 1. Data Loading ==========
ValDataPath = r'./data/Validation/images'
ValLabelPath = r'./data/Validation/labels'

val_data_files = sorted(glob.glob(os.path.join(ValDataPath, '*.mat')))[:]
val_label_files = sorted(glob.glob(os.path.join(ValLabelPath, '*.png')))[:]

val_data_list, val_labels_list, raw_images = [], [], []

for df, lf in zip(val_data_files, val_label_files):
    mat_data = sio.loadmat(df)['image']
    label = np.array(Image.open(lf))
    raw_images.append(mat_data)
    val_data_list.append(mat_data)
    val_labels_list.append(label)

val_data_tensor = torch.tensor(np.array(val_data_list)).float()
val_label_tensor = torch.tensor(np.array(val_labels_list)).long()
NC = val_data_tensor.shape[3]

val_loader = DataLoader(TensorDataset(val_data_tensor, val_label_tensor), batch_size=1)

# ========== 2. Model Loading ==========
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512", num_labels=4, ignore_mismatched_sizes=True
)

embed_dim = model.segformer.encoder.patch_embeddings[0].proj.out_channels
model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(NC, embed_dim, kernel_size=7, stride=4, padding=3)

model_path = r'./saved_models/final_production_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ========== 3. Inference & Probability Computation ==========
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for data, target in val_loader:
        data = data.permute(0, 3, 1, 2).to(device)
        logits = model(data).logits
        logits = nn.functional.interpolate(logits, size=target.shape[1:], mode='bilinear')
        pred = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        all_preds.append(pred.cpu().numpy())
        all_labels.append(target.numpy())
        all_probs.append(probs.cpu().numpy())

preds_flat = np.concatenate(all_preds).flatten()
labels_flat = np.concatenate(all_labels).flatten()

probs_all = np.concatenate(all_probs, axis=0)
probs_all = np.transpose(probs_all, (0, 2, 3, 1))
probs_flat = probs_all.reshape(-1, 4)

# ========== 4. Normalized Confusion Matrix ==========
print("Saving Confusion Matrix...")
cm_n = confusion_matrix(labels_flat, preds_flat, labels=[0, 1, 2], normalize='true')

plt.figure(figsize=(8, 6))
sns.heatmap(cm_n, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            annot_kws={"size": 18, "weight": "bold"})

plt.xticks(fontsize=14)
plt.yticks(fontsize=14, rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'Figure_Confusion_Matrix.png'), dpi=300)
plt.close()

# ========== 5. IoU Calculation ==========
print("Calculating IoU...")

iou_per_class = []
for cls_idx in range(3):
    TP = np.sum((preds_flat == cls_idx) & (labels_flat == cls_idx))
    FP = np.sum((preds_flat == cls_idx) & (labels_flat != cls_idx))
    FN = np.sum((preds_flat != cls_idx) & (labels_flat == cls_idx))
    iou = TP / (TP + FP + FN + 1e-8)
    iou_per_class.append(iou)
    print(f"  {CLASS_NAMES[cls_idx]} IoU: {iou:.4f} ({iou*100:.2f}%)")

mean_iou = np.mean(iou_per_class)
print(f"  Mean IoU (mIoU): {mean_iou:.4f} ({mean_iou*100:.2f}%)")

# IoU bar chart
plt.figure(figsize=(7, 5))
bars = plt.bar(CLASS_NAMES, [v*100 for v in iou_per_class],
               color=['#4472C4', '#EF2D0F', '#009E73'], edgecolor='black', linewidth=0.8)

for bar, val in zip(bars, iou_per_class):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val*100:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.axhline(y=mean_iou*100, color='gray', linestyle='--', linewidth=1.5,
            label=f'mIoU = {mean_iou*100:.2f}%')
plt.ylabel('IoU (%)', fontsize=15)
plt.ylim(0, 105)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'Figure_IoU.png'), dpi=300)
plt.close()
print("IoU figure saved!")

# ========== 6. F1 Curve & PR Curve ==========
print("Saving F1 Curve and PR Curve...")
target_classes = [(1, 'Eosinophil'), (2, 'Edema')]
f1_data, pr_data = {}, {}

for i, (cls_idx, cls_name) in enumerate(target_classes):
    y_true_binary = (labels_flat == cls_idx).astype(int)
    y_scores = probs_flat[:, cls_idx]
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    f1_data[cls_name] = (thresholds, f1_scores[:-1])
    pr_data[cls_name] = (recall, precision, auc(recall, precision))

# F1 Curve
plt.figure(figsize=(8, 6))
for cls_name, (thresh, f1) in f1_data.items():
    plt.plot(thresh, f1, label=cls_name, linewidth=3)

plt.title("F1-Score vs. Confidence Threshold", fontsize=18, pad=15)
plt.xlabel("Confidence (Probability)", fontsize=15)
plt.ylabel("F1-Score", fontsize=15)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower left', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'Figure_F1_Curve.png'), dpi=300)
plt.close()

# PR Curve
plt.figure(figsize=(8, 6))
for cls_name, (rec, prec, auc_val) in pr_data.items():
    plt.plot(rec, prec, label=f'{cls_name} (AUC={auc_val:.3f})', linewidth=3)

plt.title("Precision - Recall Curve", fontsize=18, pad=15)
plt.xlabel("Recall", fontsize=18)
plt.ylabel("Precision", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower left', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'Figure_PR_Curve.png'), dpi=300)
plt.close()

# ========== 7. Prediction Images (pixel-only, no margins) ==========
print("Saving GT and Prediction images...")
# 0: white (Background), 1: red (Eosinophil), 2: green (Edema)
custom_cmap = ListedColormap(['white', '#EF2D0F', '#009E73'])

for i in range(len(val_data_files)):
    plt.imsave(os.path.join(SAVE_DIR, f'Img_{i+1}_GT.png'), val_labels_list[i], cmap=custom_cmap, vmin=0, vmax=2)
    plt.imsave(os.path.join(SAVE_DIR, f'Img_{i+1}_Pred.png'), all_preds[i][0], cmap=custom_cmap, vmin=0, vmax=2)

# ========== 8. Legend Figure ==========
print("Saving legend figure...")
fig_leg = plt.figure(figsize=(3, 1.5))
ax_leg = fig_leg.add_subplot(111)
ax_leg.axis('off')

leg_bg = mpatches.Patch(color='white', ec='black', label='Background')
leg_eos = mpatches.Patch(color='#EF2D0F', label='Eosinophil')
leg_edema = mpatches.Patch(color='#009E73', label='Edema')

ax_leg.legend(handles=[leg_bg, leg_eos, leg_edema], loc='center', fontsize=12, frameon=False)
plt.savefig(os.path.join(SAVE_DIR, 'Figure_Legend.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"All figures saved to '{SAVE_DIR}' directory.")