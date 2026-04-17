import yaml
import numpy as np
import os
import glob
import shutil
import torch
import tifffile as tiff
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from PIL import Image

class YOLOproject:
    """
    A comprehensive wrapper class for YOLOv12 training and dataset management.
    Supports multi-channel TIFF images and automated dataset splitting.
    """
    def __init__(self, img_Path, label_Path, extension='*.*', model='yolo12n.pt'):
        """
        Args:
            img_Path (str): Path to directory containing images.
            label_Path (str): Path to directory containing labels (.txt).
            extension (str): Image file extension pattern.
            model (str): Pre-trained YOLO model weight path.
        """
        self.img_dir = img_Path
        self.label_dir = label_Path
        self.extension = extension
        
        self.Imglist, self.Labellist = self.load_paths()
        self.Model = YOLO(model)
        
        # Metadata extraction from sample
        sample = self.__getitem__(0)
        self.model_name = os.path.splitext(model)[0]
        self.channels = sample.shape[2] if sample.ndim == 3 else 1
        self.img_size = sample.shape[1]
        
    def load_paths(self):
        """Loads and synchronizes image and label paths."""
        imgs = sorted(glob.glob(os.path.join(self.img_dir, self.extension)))
        labels = sorted(glob.glob(os.path.join(self.label_dir, '*.txt')))
        
        print(f"Total Images: {len(imgs)}, Total Labels: {len(labels)}")
        assert len(imgs) == len(labels), "Error: Mismatch between image and label counts."
        return imgs, labels
     
    def __getitem__(self, idx):
        """Reads image based on file extension (Supports TIFF and standard formats)."""
        img_path = self.Imglist[idx]
        try:
            if img_path.lower().endswith(('.tiff', '.tif')):
                # Multi-channel TIFF handling (C, H, W -> H, W, C)
                img = np.transpose(tiff.imread(img_path), (1, 2, 0))
            else:
                img = np.array(Image.open(img_path))
            return img
        except Exception as e:
            print(f"Error loading image at {img_path}: {e}")
            return None
    
    def split_dataset(self, Val_size=0.2):
        """
        Splits dataset into Train and Validation sets and organizes folder structure.
        Args:
            Val_size (float): Ratio of the validation set.
        """
        train_val, val_set = train_test_split(
            self.Imglist, 
            test_size=Val_size, 
            random_state=42
        )                   
        
        print(f"Dataset Split: Train({len(train_val)}) | Val({len(val_set)})")
        self._copy_files(train_val, 'train')
        self._copy_files(val_set, 'val')
        
        return train_val, val_set
    
    def _copy_files(self, file_list, split_name='train'):
        """Internal helper to copy images and labels to split directories."""
        target_img_dir = os.path.join(self.img_dir, split_name, 'images')
        target_label_dir = os.path.join(self.img_dir, split_name, 'labels')
        
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_label_dir, exist_ok=True)
        
        for img_path in file_list:
            # Copy image
            shutil.copy(img_path, target_img_dir)
            
            # Copy corresponding label
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(self.label_dir, label_name)
            
            if os.path.exists(label_path):
                shutil.copy(label_path, target_label_dir)
        
    def create_Full_yaml(self, output_path, classes):
        """Generates YOLO configuration YAML file with customized data augmentation."""
        yaml_content = {
            'train': os.path.abspath(os.path.join(self.img_dir, 'train', 'images')),
            'val': os.path.abspath(os.path.join(self.img_dir, 'val', 'images')),
            'channels': self.channels,
            'nc': len(classes),
            'names': classes,
            # Data Augmentation Parameters for robust medical imaging detection
            'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
            'degrees': 15.0, 'translate': 0.1, 'scale': 0.5, 'shear': 5.0,
            'flipud': 0.5, 'fliplr': 0.5, 'mixup': 0.15, 'mosaic': 0.5
        }
        with open(output_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        return output_path
        
    def Train_Model(self, data_yaml, epochs=200, project='YOLO_Project', Val=True):
        """Executes YOLO model training with optimized hyperparameters."""
        try:
            results = self.Model.train(
                data=data_yaml,
                epochs=epochs,
                batch=8,
                imgsz=self.img_size,
                device=0 if torch.cuda.is_available() else 'cpu',
                project=project,
                name=f'{self.model_name}_Model',
                exist_ok=True,
                patience=50,
                # Optimization Settings
                lr0=0.0001, lrf=0.01,
                warmup_epochs=15,
                box=7.5, cls=0.7, dfl=1.0,
                save=True,
                val=Val,
                augment=True
            )
            return results
        except KeyboardInterrupt:
            print("Training interrupted by user.")
            return None
        
    def __len__(self):
        return len(self.Imglist)