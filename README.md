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


### 1. Requirements
pip install -r requirements.txt

### 2. Run Inference
python predict.py --input ./data/sample.png --weights ./models/best.pt

python image_process.py --input ./raw_data --output ./processed_data --normalize

python yolo_train.py --config config/yolo_cfg.yaml --epochs 100

# Performs patch-wise inference and generates density maps
python patch_prediction.py --input ./test_images --output ./results

python hyperspectral_viewer.py --file ./data/sample.hsi
