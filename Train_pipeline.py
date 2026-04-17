"""
YOLOv12 Training Pipeline for Eosinophil Detection
Author: Jin-Il-Jang
Description:
    Automated pipeline for dataset splitting, YAML configuration generation, 
    and model training using a custom YOLO module.
"""

import argparse
import os
import YOLO_Module as YOD

def main(args):
    # 1. Initialize YOLO Project
    # We use argparse to handle paths instead of hardcoded local Windows paths.
    plastic_yolo = YOD.YOLOproject(
        data_path=args.data_path,
        label_path=args.label_path,
        model=args.model
    )

    # 2. Dataset Split (Train/Val)
    print(f"Splitting dataset with validation size: {args.val_size}")
    train, val = plastic_yolo.split_dataset(Val_size=args.val_size)

    # 3. YAML Configuration Generation
    # Generating YAML for the specified classes
    class_names = ['Eosinophil', 'Nucleus']
    yaml_path = plastic_yolo.create_Full_yaml(args.yaml_name, class_names)
    print(f"YAML configuration created at: {yaml_path}")

    # 4. Model Training
    plastic_yolo.Train_Model(
        yaml_path=yaml_path, 
        epochs=args.epochs, 
        project=args.project_name, 
        Val=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Training Pipeline")
    
    # Path Arguments
    parser.add_argument('--data_path', type=str, required=True, help="Directory containing raw images")
    parser.add_argument('--label_path', type=str, required=True, help="Directory containing YOLO labels")
    
    # Hyperparameters
    parser.add_argument('--model', type=str, default='yolo12n.pt', help="Pre-trained weights (e.g., yolo12n.pt)")
    parser.add_argument('--val_size', type=float, default=0.2, help="Validation set ratio")
    parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs")
    parser.add_argument('--yaml_name', type=str, default='eosinophil_config.yaml', help="Output YAML filename")
    parser.add_argument('--project_name', type=str, default='Eosinophil_Project', help="Project name for logging")

    args = parser.parse_args()
    main(args)