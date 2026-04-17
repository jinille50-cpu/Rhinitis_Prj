# Hybrid YOLO-Segformer Pipeline for Precision Vision Analysis

YOLO, Segmenation, Image process
# Department of Otorhinolaryngology-Head and Neck Surgery, Boramae Medical Center
# Kookmin University

Auto-GT Generation

Language: Python 3.10.11

Framework: PyTorch

Model: YOLOv12, Segformer, Transformer-based architectures

Libraries: OpenCV, NumPy, Scikit-learn, Matplotlib


This project is designed with a modular architecture for scalability and ease of maintenance. Each module handles a specific part of the vision pipeline.

### 1. Core Engine
- `yolo_train.py`: Main script for training the YOLO detection model.
- `yolo_module.py`: Configuration and parameter management for YOLO architectures.
- `eosin_prediction.py`: Inference module for Eosinophil detection using trained YOLO weights.
- `patch_prediction.py`: Automated sliding-window inference logic for large-scale patch processing.

### 2. Models (Segmentation)
- `models/segmentation/`: Directory containing various segmentation architectures (e.g., Segformer, U-Net 3+).
- `density_image_process.py`: Post-processing module that converts segmentation maps into density heatmaps.

### 3. Data Processing & Utilities
- `image_process.py`: Image preprocessing suite (Crop, Data Normalization, Augmentation).
- `file_utils.py`: File system management and automation for batch processing.
- `label_editor.py`: Utility for refining and correcting Ground Truth labels.

### 4. Specialized Analysis
- `hyperspectral_viewer.py`: Interactive GUI tool for analyzing fluorescence hyperspectral data cubes.

---



# Performs patch-wise inference and generates density maps
### Density Heatmap Generation
This module visualizes the spatial distribution of Eosinophils by generating heatmaps from grid-based prediction data.

**Key Features:**
- **Absolute Scaling:** Ensures consistent visualization across multiple samples (Default range: 3 to 30 counts).
- **Alpha Blending:** Overlays heatmap onto original H&E images with adjustable transparency.
- **Colorbar Integration:** Generates publication-ready figures with calibrated color scales.

**How to Run:**

How to Run & Interactive Tools

This project provides a complete pipeline from raw image processing to deep learning training and interactive visualization.

### 1. Core Pipelines (CLI)
Execute the following commands in your terminal to process data or train the model.

# Data Preparation (Patch Processing)
python "Image Process.py" --input ./raw_data/sample.jpg --output ./data/processed --base_name B_0326

# Run the end-to-end training pipeline including dataset splitting and YAML configuration.
python Train_pipeline.py --data_path ./data/processed --label_path ./data/labels --epochs 300

# Density Map Generation (Post-Analysis)
python "Density Image process.py" --image ./result/sample_scaled.jpg --vmin 3 --vmax 30


python "YOLO Eosin LabelEditer.py"
# Yolo Label Editer
<img width="1402" height="966" alt="image" src="https://github.com/user-attachments/assets/e2d6d1d1-01e0-48df-badf-88bcc54273e8" />


python "Hyperspectral Viewer.py" --file ./data/sample.hsi
# Hyperspectral Viewer
<img width="1602" height="966" alt="image" src="https://github.com/user-attachments/assets/1a51b074-ccdc-4d7f-ac63-2ba31aa80069" />


