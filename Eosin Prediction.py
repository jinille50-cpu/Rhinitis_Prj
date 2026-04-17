import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile
import tkinter as tk
from tkinter import filedialog
import os
from ultralytics import YOLO

def load_image_rgb(image_path):
    """Load image and return in RGB format"""
    if image_path.lower().endswith(('.tif', '.tiff')):
        image = tifffile.imread(image_path)
        # Handle cases where TIFF might be loaded in different channel orders
        if image.ndim == 3 and image.shape[0] < image.shape[2]:
            image = np.transpose(image, (1, 2, 0))
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def main():
    # --- Configuration ---
    # Relative path for model portability
    model_path = os.path.join('models', 'best.pt') 
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please place your 'best.pt' inside a 'models' folder.")
        
        # Fallback to file dialog if default path fails
        root = tk.Tk(); root.withdraw()
        model_path = filedialog.askopenfilename(title='Select YOLO Model (best.pt)', 
                                              filetypes=[('PyTorch Model', '*.pt')])
        root.destroy()
        if not model_path: return

    # Load YOLO Model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # --- Image Selection ---
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    image_path = filedialog.askopenfilename(
        title='Select Image for Inference',
        filetypes=[('Image files', '*.mat *.h5 *.hdf5 *.tiff *.tif *.jpg *.png'), ('All files', '*.*')]
    )
    root.destroy()
    
    if not image_path:
        print("No image selected. Exiting...")
        return
    
    # Load and Prepare Image
    image_rgb = load_image_rgb(image_path)
    
    # Convert to BGR for YOLO inference
    img_for_yolo = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Run Prediction
    # Setting conf and iou consistent with your MSILabelerRGB app
    results = model.predict(img_for_yolo, conf=0.3, iou=0.45, imgsz=512, verbose=False)
    
    # --- Visualization ---
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image_rgb)
    
    class_names = {0: 'Eosinophil', 1: 'Nucleus'}
    class_colors = {0: 'red', 1: 'blue'}
    
    count_0 = 0
    count_1 = 0

    # Draw Bounding Boxes
    if results[0].boxes is not None:
        for box in results[0].boxes:
            # Extract coordinates and info
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            # Update Counters
            if cls == 0: count_0 += 1
            elif cls == 1: count_1 += 1
            
            # Set Label and Color
            color = class_colors.get(cls, 'green')
            name = class_names.get(cls, f'Class {cls}')
            
            # Create Rectangle Patch
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                   linewidth=1.5, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Add Text Label
            ax.text(x1, y1 - 5, f'{name} {conf:.2f}', 
                    color=color, fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
            
    # Finalize Plot
    ax.axis('off')
    filename = os.path.basename(image_path)
    plt.title(f"File: {filename}\nDetect: Eosin({count_0}) | Nucleus({count_1})", 
              fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the result automatically
    save_name = f"result_{filename.split('.')[0]}.png"
    plt.savefig(save_name, dpi=300)
    print(f"Inference complete. Result saved as {save_name}")
    
    plt.show()

if __name__ == "__main__":
    main()