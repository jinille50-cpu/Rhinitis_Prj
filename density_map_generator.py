import numpy as np
import cv2
import os
import time
import tifffile
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from ultralytics import YOLO

def create_output_folder():
    """Create a 'result' folder in the current working directory"""
    current_dir = os.getcwd()
    output_folder = os.path.join(current_dir, "result")
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def load_large_image(image_path):
    """Load large-scale images (TIFF supported)"""
    try:
        image = tifffile.imread(image_path)
        # Convert RGB to BGR for OpenCV consistency if needed
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def crop_image_with_overlap(image, crop_size=512, overlap_ratio=0.25):
    """Generate patches with a specified overlap ratio to handle boundary issues"""
    height, width = image.shape[:2]
    stride = int(crop_size * (1 - overlap_ratio)) 
    
    patches = []
    patch_info = [] # Stores (y1, x1) global coordinates
    
    y_steps = list(range(0, height - crop_size + 1, stride))
    if y_steps and y_steps[-1] + crop_size < height:
        y_steps.append(height - crop_size)
        
    x_steps = list(range(0, width - crop_size + 1, stride))
    if x_steps and x_steps[-1] + crop_size < width:
        x_steps.append(width - crop_size)

    print(f"Image Size: {height} x {width}")
    print(f"Overlap Ratio: {overlap_ratio*100}% (Stride: {stride}px)")
    print(f"Total Expected Patches: {len(y_steps) * len(x_steps)}")

    for y1 in y_steps:
        for x1 in x_steps:
            y2, x2 = y1 + crop_size, x1 + crop_size
            patch = image[y1:y2, x1:x2]
            
            if patch.shape[0] == crop_size and patch.shape[1] == crop_size:
                patches.append(patch)
                patch_info.append((y1, x1))

    return patches, patch_info

def batch_predict_patches(model, patches, conf=0.4, batch_size=32):
    """Predict patches in batches for better GPU utilization"""
    results = []
    total = len(patches)
    print(f"Processing {total} patches with batch size {batch_size}...")
    
    for i in range(0, total, batch_size):
        batch_end = min(i + batch_size, total)
        batch_patches = patches[i:batch_end]
        
        try:
            # device=0 attempts to use GPU. Change to 'cpu' if no GPU available.
            batch_results = model.predict(
                batch_patches, conf=conf, verbose=False, device=0,
                imgsz=512, max_det=300, save=False, show=False, classes=0
            )
            results.extend(batch_results)
        except Exception as e:
            print(f"Batch {i//batch_size + 1} failed, attempting individual processing: {e}")
            for patch in batch_patches:
                try:
                    res = model.predict(patch, conf=conf, verbose=False)[0]
                    results.append(res)
                except:
                    results.append(None)
    return results

def merge_overlapping_predictions(patch_info, results, iou_threshold=0.3):
    """Merge duplicate boxes across overlapping patches using Global NMS"""
    print("Performing Global Non-Maximum Suppression (NMS)...")
    
    global_boxes = []
    global_confs = []
    
    for (y1, x1), result in zip(patch_info, results):
        if result is None or result.boxes is None: continue
        
        for box in result.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            
            # Convert patch coordinates to global image coordinates
            gx1, gy1 = bx1 + x1, by1 + y1
            gx2, gy2 = bx2 + x1, by2 + y1
            
            w, h = gx2 - gx1, gy2 - gy1
            global_boxes.append([int(gx1), int(gy1), int(w), int(h)])
            global_confs.append(conf)

    if not global_boxes:
        return []

    # OpenCV NMS to remove duplicates
    indices = cv2.dnn.NMSBoxes(global_boxes, global_confs, score_threshold=0.1, nms_threshold=iou_threshold)
    
    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            gx, gy, gw, gh = global_boxes[i]
            conf = global_confs[i]
            final_boxes.append([gx, gy, gx+gw, gy+gh, conf])
            
    print(f"-> Before NMS: {len(global_boxes)} | After NMS: {len(final_boxes)} (Actual Objects)")
    return final_boxes

def create_density_grid(final_boxes, img_height, img_width, grid_size=512):
    """Calculate object counts within a fixed grid"""
    rows = img_height // grid_size
    cols = img_width // grid_size
    grid_matrix = np.zeros((rows, cols), dtype=int)
    
    for box in final_boxes:
        gx1, gy1, gx2, gy2, conf = box
        cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
        
        grid_x = int(cx // grid_size)
        grid_y = int(cy // grid_size)
        
        if grid_y < rows and grid_x < cols:
            grid_matrix[grid_y, grid_x] += 1
            
    return grid_matrix, rows, cols

def save_results(grid_matrix, output_folder, image_name):
    """Save density map and summary to text files"""
    rows, cols = grid_matrix.shape
    total_count = np.sum(grid_matrix)
    
    # Save Grid data (TSV format for Excel)
    grid_file = os.path.join(output_folder, f"{image_name}_grid.txt")
    np.savetxt(grid_file, grid_matrix, fmt='%d', delimiter='\t')
            
    # Save Summary
    summary_file = os.path.join(output_folder, f"{image_name}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"=== Detection Summary: {image_name} ===\n")
        f.write(f"Grid Dimensions: {rows} x {cols}\n")
        f.write(f"Total Detected Objects: {total_count}\n")

    print(f"Results saved to: {output_folder}")
    print(f"Total count: {total_count}")

def create_scaled_composite(original_image, output_folder, image_name, scale_factor=0.1):
    """Save a downscaled version of the original image for reference"""
    print(f"Generating downscaled preview (Scale: {scale_factor})...")
    orig_height, orig_width = original_image.shape[:2]
    final_height = int(orig_height * scale_factor)
    final_width = int(orig_width * scale_factor)
    
    composite_image = cv2.resize(original_image, (final_width, final_height))
    composite_path = os.path.join(output_folder, f"{image_name}_preview.jpg")
    cv2.imwrite(composite_path, composite_image)
    print(f"Preview saved: {composite_path}")

def main():
    # --- Config ---
    model_path = os.path.join('models', 'best.pt') 
    conf_threshold = 0.4
    overlap_ratio = 0.25 
    
    print("=== Large-scale Image Object Detection (YOLO) ===")
    
    # File Selection
    root = tk.Tk(); root.withdraw()
    image_path = filedialog.askopenfilename(title='Select Large Image File')
    root.destroy()
    
    if not image_path: return
    
    # Check Model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please check your path.")
        return

    output_folder = create_output_folder()
    
    print("Loading model and image...")
    model = YOLO(model_path)
    image = load_large_image(image_path)
    if image is None: return
    image_name = Path(image_path).stem
    
    start_time = time.time()
    
    # 1. Tiling with Overlap
    patches, patch_info = crop_image_with_overlap(image, 512, overlap_ratio)
    
    # 2. Batch Inference
    results = batch_predict_patches(model, patches, conf=conf_threshold, batch_size=32)
    
    # 3. Coordinate Transformation & Global NMS
    final_boxes = merge_overlapping_predictions(patch_info, results, iou_threshold=0.3)
    
    # 4. Density Mapping
    grid_matrix, rows, cols = create_density_grid(final_boxes, image.shape[0], image.shape[1], 512)
    
    # 5. Export Results
    save_results(grid_matrix, output_folder, image_name)
    create_scaled_composite(image, output_folder, image_name, scale_factor=0.1)

    total_time = time.time() - start_time
    print(f"\nTotal Processing Time: {total_time:.1f} seconds")

if __name__ == "__main__":
    main()