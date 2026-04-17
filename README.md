# Hybrid YOLO-Segformer Pipeline for Precision Vision Analysis
**Keywords:** YOLO, Segmentation, Image Processing | **Auto-GT Generation**

### 🏥 Affiliations
* **Department of Otorhinolaryngology-Head and Neck Surgery, Boramae Medical Center**
* **Kookmin University**

---

### 🛠 Tech Stack
* **Language:** Python 3.10.11
* **Framework:** PyTorch
* **Models:** YOLOv12, Segformer, Transformer-based architectures
* **Libraries:** OpenCV, NumPy, Scikit-learn, Matplotlib

---
### Requirements

torch==2.8.0+cu129
numpy==1.24.3
opencv-python==4.11.0.86
matplotlib==3.9.0
scikit-learn==1.5.0
tifffile==2025.3.30
scipy==1.14.0
ultralytics==8.3.153


## 🏗 Project Architecture
This project is designed with a modular architecture for scalability and ease of maintenance. Each module handles a specific part of the vision pipeline.

**1. Core Engine**
* `yolo_train.py`: Main script for training the YOLO detection model.
* `yolo_module.py`: Configuration and parameter management for YOLO architectures.
* `eosin_prediction.py`: Inference module for Eosinophil detection using trained YOLO weights.
* `patch_prediction.py`: Automated sliding-window inference logic for large-scale patch processing.

**2. Models (Segmentation)**
* `models/segmentation/`: Directory containing various segmentation architectures (e.g., Segformer, U-Net 3+).
* `density_image_process.py`: Post-processing module that converts segmentation maps into density heatmaps.

**3. Data Processing & Utilities**
* `image_process.py`: Image preprocessing suite (Crop, Data Normalization, Augmentation).
* `file_utils.py`: File system management and automation for batch processing.
* `label_editor.py`: Utility for refining and correcting Ground Truth labels.

**4. Specialized Analysis**
* `hyperspectral_viewer.py`: Interactive GUI tool for analyzing fluorescence hyperspectral data cubes.

---

## ✨ Key Features: Density Heatmap Generation
Performs patch-wise inference and visualizes the spatial distribution of Eosinophils by generating heatmaps from grid-based prediction data.
* **Absolute Scaling:** Ensures consistent visualization across multiple samples (Default range: 3 to 30 counts).
* **Alpha Blending:** Overlays heatmap onto original H&E images with adjustable transparency.
* **Colorbar Integration:** Generates publication-ready figures with calibrated color scales.

---

## 🚀 How to Run & Interactive Tools
This project provides a complete pipeline from raw image processing to deep learning training and interactive visualization.

## 1. Core Pipelines (CLI)
Execute the following commands in your terminal to process data or train the model.

**Data Preparation (Patch Processing)**
```bash
python "Image Process.py" --input ./raw_data/sample.jpg --output ./data/processed --base_name Sample_Day
```
## Run the end-to-end training pipeline including dataset splitting and YAML configuration.
```bash
python Train_pipeline.py --data_path ./data/processed --label_path ./data/labels --epochs 300
```
## Density Map Generation (Post-Analysis)
### Step 1: Generate density grid data (Patch-wise inference with Global NMS)
```bash
python density_map_generator.py
```
### Step 2: Generate Density Heatmap (Post-analysis visualization)
```bash
python "Density Image process.py" --image ./result/sample_preview.jpg --vmin 3 --vmax 30
```

## 2_1 Interactive GUI Tools (Object Detection Labeler)
```bash
python "YOLO Eosin LabelEditer.py"
```

## 2_2 Interactive GUI Tools (Hyperspectral Viewer)
```bash
python "YOLO Eosin Hyperspectral Viewer.py"
```
# Yolo Label Editer
<img width="1402" height="966" alt="image" src="https://github.com/user-attachments/assets/e2d6d1d1-01e0-48df-badf-88bcc54273e8" />


python "Hyperspectral Viewer.py" --file ./data/sample.hsi
# Hyperspectral Viewer
<img width="1602" height="966" alt="image" src="https://github.com/user-attachments/assets/ee106ecd-86a6-4083-b4d4-8dbab024e749" />



