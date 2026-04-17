"""
Image Patch Processor: Automatic Cropping & Normalization
Author: Jin-Il-Jang
Description:
    This script splits large histology images into 512x512 patches, 
    applies individual normalization, and saves them with sequential indexing.
    Includes a safety feature to prevent overwriting existing files in the target directory.
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import File_Util as Fu  # Custom utility for image normalization

def get_next_index(save_dir, base_name):
    """Finds the highest existing index in the directory to prevent overwriting."""
    max_idx = 0
    if not os.path.exists(save_dir):
        return 0
        
    for fname in os.listdir(save_dir):
        if fname.startswith(f"{base_name}_") and fname.endswith(".png"):
            try:
                stem = os.path.splitext(fname)[0]
                idx_str = stem.split("_")[-1]
                max_idx = max(max_idx, int(idx_str))
            except (ValueError, IndexError):
                continue
    return max_idx

def process_patches(image_path, target_folder, base_name, crop_size=512):
    """Crops images into patches and applies normalization."""
    # Ensure save directory exists
    os.makedirs(target_folder, exist_ok=True)
    
    # Read image using numpy for Unicode path support (Korean characters)
    img_array = np.fromfile(image_path, np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        print(f"Error: Could not load image from {image_path}")
        return

    h, w, _ = img_bgr.shape
    rows, cols = h // crop_size, w // crop_size
    
    current_index = get_next_index(target_folder, base_name)
    print(f"Safety Check: Starting from index {current_index + 1} for '{base_name}'")
    
    saved_count = 0
    for i in range(rows):
        for j in range(cols):
            y1, y2 = i * crop_size, (i + 1) * crop_size
            x1, x2 = j * crop_size, (j + 1) * crop_size
            
            patch_bgr = img_bgr[y1:y2, x1:x2]
            
            if patch_bgr.shape[0] == crop_size and patch_bgr.shape[1] == crop_size:
                # 1. Individual Patch Normalization
                patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
                # Assuming Fu.normalize_image exists in your File_Util.py
                patch_norm_rgb = Fu.normalize_image(patch_rgb)
                patch_final_bgr = cv2.cvtColor(patch_norm_rgb, cv2.COLOR_RGB2BGR)
                
                # 2. Sequential Indexing & Saving
                current_index += 1 
                file_name = f"{base_name}_{current_index:04d}.png"
                save_path = os.path.join(target_folder, file_name)
                
                is_success, buffer = cv2.imencode(".png", patch_final_bgr)
                if is_success:
                    with open(save_path, "wb") as f:
                        buffer.tofile(f)
                    saved_count += 1
            
            if saved_count % 500 == 0 and saved_count > 0:
                print(f"Progress: {saved_count} patches processed...")

    print(f"\nProcessing Complete! Total new patches saved: {saved_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Patch Cropper & Normalizer")
    
    # Path Arguments
    parser.add_argument('--input', type=str, help="Path to the input large image")
    parser.add_argument('--output', type=str, default="./processed_patches", help="Target save directory")
    parser.add_argument('--base_name', type=str, default="Sample", help="Base name for file naming")
    parser.add_argument('--crop_size', type=int, default=512, help="Patch size (default: 512)")

    args = parser.parse_args()

    # GUI Fallback for ease of use in local environment
    if not args.input:
        print("No input provided via CLI. Opening file selector...")
        args.input = Fu.Uigetfile()
    
    if args.input:
        process_patches(args.input, args.output, args.base_name, args.crop_size)