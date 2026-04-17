"""
Density Heatmap Generator for Eosinophil Distribution Analysis
Author: Jin-Il-Jang
Description: 
    This script generates a density heatmap from grid-based count data 
    and overlays it on the original H&E or scaled images.
    It supports absolute scaling for consistent comparison between different samples.
"""

import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path

# Custom utility for file selection (Keep your File_Util if needed)
# In a shared environment, it's better to use argparse instead of Uigetfile

def imread_korean(filename, flags=cv2.IMREAD_COLOR):
    """Reading image files with Korean file paths using numpy"""
    try:
        img_array = np.fromfile(filename, np.uint8)
        return cv2.imdecode(img_array, flags)
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

def imwrite_korean(filename, img, params=None):
    """Writing image files with Korean file paths using numpy"""
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        return False
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def generate_density_heatmap(img_path, grid_path, output_dir, vmin=3, vmax=30, alpha_bg=0.5, alpha_hm=0.4):
    """
    Core function to process density heatmap overlay.
    Args:
        img_path (str): Path to the background scaled image.
        grid_path (str): Path to the grid text file containing counts.
        output_dir (str): Directory to save results.
        vmin/vmax (int): Absolute scale range for the heatmap.
    """
    # 1. Load Data
    background_img = imread_korean(img_path)
    if background_img is None: return
    
    bg_h, bg_w = background_img.shape[:2]
    
    try:
        grid_matrix = np.loadtxt(grid_path, delimiter='\t')
    except Exception as e:
        print(f"Failed to load grid file: {e}")
        return

    # 2. Normalize and Map Heatmap
    # Using 'YlOrRd' colormap as default for biological density
    target_cmap = plt.get_cmap('YlOrRd')
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    normalized_grid = norm(grid_matrix)
    
    heatmap_rgba = target_cmap(normalized_grid)
    heatmap_bgr_small = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)[:, :, ::-1]
    
    # Resize heatmap to match background image dimensions
    heatmap_resized = cv2.resize(heatmap_bgr_small, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)

    # 3. Alpha Blending
    final_composite = cv2.addWeighted(background_img, alpha_bg, heatmap_resized, alpha_hm, 0)

    # 4. Save Raw Composite
    base_name = Path(img_path).stem.replace("_original_scaled", "")
    os.makedirs(output_dir, exist_ok=True)
    save_path_raw = os.path.join(output_dir, f"{base_name}_density_overlay.jpg")
    imwrite_korean(save_path_raw, final_composite)

    # 5. Generate and Save Plot with Colorbar
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cv2.cvtColor(final_composite, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    
    mappable = cm.ScalarMappable(norm=norm, cmap=target_cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Eosinophil Count per Grid', rotation=270, labelpad=20, fontsize=12)
    
    save_path_fig = os.path.join(output_dir, f"{base_name}_density_with_colorbar.png")
    plt.savefig(save_path_fig, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Processing complete for: {base_name}")
    print(f"Saved at: {output_dir}")

if __name__ == "__main__":
    # Command line argument parsing for professional CLI usage
    parser = argparse.ArgumentParser(description="Generate Density Heatmaps from Grid Data")
    parser.add_argument('--image', type=str, required=True, help="Path to the scaled background image")
    parser.add_argument('--grid', type=str, help="Path to the grid .txt file (auto-detects if omitted)")
    parser.add_argument('--output', type=str, default="./output_density_maps", help="Output directory")
    parser.add_argument('--vmin', type=int, default=3, help="Minimum count for scale")
    parser.add_argument('--vmax', type=int, default=30, help="Maximum count for scale")

    args = parser.parse_args()

    # Auto-detect grid file if not provided
    if not args.grid:
        base_name = Path(args.image).stem.replace("_original_scaled", "")
        args.grid = os.path.join(os.path.dirname(args.image), f"{base_name}_grid.txt")

    generate_density_heatmap(args.image, args.grid, args.output, args.vmin, args.vmax)