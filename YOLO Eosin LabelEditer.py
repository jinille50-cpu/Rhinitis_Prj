"""
YOLO Eosin Label Editor
Author: Jin-Il-Jang
Description:
    An interactive GUI tool for editing YOLO labels. 
    Supports automated pre-labeling by integrating a trained YOLO model,
    manual box drawing, and sequential class assignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
import tkinter as tk
from tkinter import filedialog
import os
import cv2
import argparse
from ultralytics import YOLO
import File_Util as Fu 

class MSILabelerRGB:
    def __init__(self, model_path=None, init_path=None):
        # Configuration
        self.default_model_path = model_path if model_path else 'yolo12s.pt'
        self.initpath = init_path if init_path else os.getcwd()
        
        self.class_names = {0: 'Eosin', 1: 'Nucleus'}  
        self.class_colors = {0: 'red', 1: 'blue'}
        
        # State Variables
        self.data = None       
        self.model = None
        self.boxes = []        
        self.selected_idx = None
        self.brightness = 1.0
        self.loaded_filename = ""
        self.conf_threshold = 0.3  
        
        self.is_drawing = False
        self.is_resizing = False
        self.start_pos = None
        
        # UI Setup
        self.fig = plt.figure(figsize=(14, 9))
        self.ax = self.fig.add_axes([0.05, 0.1, 0.75, 0.85])
        self.ax.axis('off')
        
        # Legend Display
        self.ax_legend = self.fig.add_axes([0.82, 0.73, 0.15, 0.2])
        self.update_legend()
        
        # UI Components: Confidence, Brightness, Buttons
        self._setup_widgets()

        # Initial Model Load
        self.auto_load_model()

        # Event Connections
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def _setup_widgets(self):
        # Confidence Box
        ax_conf = self.fig.add_axes([0.82, 0.67, 0.15, 0.04])
        self.text_conf = TextBox(ax_conf, 'Conf:', initial=str(self.conf_threshold))
        self.text_conf.on_submit(self.submit_threshold) 

        # Brightness Slider
        ax_bright = self.fig.add_axes([0.82, 0.60, 0.15, 0.03])
        self.slider_bright = Slider(ax_bright, 'Bright', 0.1, 3.0, valinit=1.0)
        self.slider_bright.on_changed(self.update_brightness)

        # Action Buttons
        ax_load_img = self.fig.add_axes([0.82, 0.50, 0.15, 0.05])
        self.btn_img = Button(ax_load_img, 'Open Image')
        self.btn_img.on_clicked(self.load_image)

        ax_load_label = self.fig.add_axes([0.82, 0.40, 0.15, 0.05])
        self.btn_label = Button(ax_load_label, 'Load Label (.txt)')
        self.btn_label.on_clicked(self.manual_load_label)

        ax_load_model = self.fig.add_axes([0.82, 0.30, 0.15, 0.05])
        self.btn_model = Button(ax_load_model, 'Load Model')
        self.btn_model.on_clicked(self.manual_load_model)

        ax_save = self.fig.add_axes([0.82, 0.15, 0.15, 0.08], facecolor='honeydew')
        self.btn_save = Button(ax_save, 'SAVE TXT\n(YOLO)')
        self.btn_save.on_clicked(self.save_labels)

    def update_legend(self):
        self.ax_legend.clear()
        self.ax_legend.set_title("[ Classes ]", fontsize=10, fontweight='bold')
        self.ax_legend.axis('off')
        for i, name in self.class_names.items():
            self.ax_legend.text(0.1, 0.8 - (i*0.15), f"{i}: {name}", 
                               color=self.class_colors.get(i, 'black'), 
                               fontweight='bold', fontsize=12)

    def submit_threshold(self, text):
        try:
            new_conf = float(text)
            if 0.0 < new_conf <= 1.0:
                self.conf_threshold = new_conf
                print(f"Confidence threshold updated to: {self.conf_threshold}")
                if self.data is not None and self.model is not None:
                    self.run_prediction()
                    self.update_display()
            else:
                self.text_conf.set_val(str(self.conf_threshold))
        except ValueError:
            self.text_conf.set_val(str(self.conf_threshold)) 

    def auto_load_model(self):
        if os.path.exists(self.default_model_path):
            try:
                self.model = YOLO(self.default_model_path)
                print(f"Model loaded: {os.path.basename(self.default_model_path)}")
            except Exception as e:
                print(f"Failed to load model: {e}")

    def manual_load_model(self, event):
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(title="Select YOLO Model (.pt)", filetypes=[('YOLO model', '*.pt')])
        if path:
            self.model = YOLO(path)
            print(f"New model loaded: {path}")
        root.destroy()

    def load_image(self, event):
        path = Fu.Uigetfile(self.initpath)
        if not path: return

        self.loaded_filename = path
        img_array = np.fromfile(path, np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img_bgr is None: return

        self.data = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.model: 
            self.run_prediction()
        else:
            self.boxes = []
            
        self.update_display()
        plt.suptitle(f"File: {os.path.basename(path)}", fontsize=12)

    def run_prediction(self):
        img_for_yolo = cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR)
        results = self.model.predict(img_for_yolo, conf=self.conf_threshold, iou=0.45, verbose=False)
        
        self.boxes = []
        if results[0].boxes:
            for b in results[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                cls = int(b.cls[0].cpu().numpy())
                self.boxes.append([x1, y1, x2, y2, cls])
        print(f"Prediction done: {len(self.boxes)} objects detected.")

    def update_display(self):
        if self.data is None: return
        self.ax.clear()
        
        img_display = np.clip(self.data.astype(np.float32) * self.brightness, 0, 255).astype(np.uint8)
        self.ax.imshow(img_display)
        
        for i, (x1, y1, x2, y2, cls) in enumerate(self.boxes):
            is_sel = (i == self.selected_idx)
            color = self.class_colors.get(cls, 'red')
            edge_color = 'yellow' if is_sel else color
            lw = 3 if is_sel else 1.5
            
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            
            rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                               fill=False, edgecolor=edge_color, linewidth=lw)
            self.ax.add_patch(rect)
            
            if is_sel or len(self.boxes) < 50:
                self.ax.text(xmin, ymin-5, f"{cls}", 
                           color=edge_color, fontsize=10, fontweight='bold',
                           bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
        
        self.ax.axis('off')
        self.fig.canvas.draw_idle()

    # Interaction Handlers (Simplified)
    def on_press(self, event):
        if event.inaxes != self.ax: return
        if event.button == 3: # Right Click to Select
            self.selected_idx = None
            for i, (x1, y1, x2, y2, cls) in enumerate(self.boxes):
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                if xmin < event.xdata < xmax and ymin < event.ydata < ymax:
                    self.selected_idx = i
                    break
        elif event.button == 1: # Left Click to Draw/Resize
            if self.selected_idx is not None:
                x1, y1, x2, y2, cls = self.boxes[self.selected_idx]
                if abs(event.xdata - max(x1,x2)) < 30 and abs(event.ydata - max(y1,y2)) < 30:
                    self.is_resizing = True
                    return
            self.is_drawing = True
            self.boxes.append([event.xdata, event.ydata, event.xdata, event.ydata, 0])
            self.selected_idx = len(self.boxes) - 1
        self.update_display()

    def on_motion(self, event):
        if event.inaxes != self.ax or event.xdata is None: return
        if self.is_drawing or self.is_resizing:
            self.boxes[self.selected_idx][2] = event.xdata
            self.boxes[self.selected_idx][3] = event.ydata
            self.update_display()

    def on_release(self, event):
        if self.is_drawing and self.boxes:
            x1, y1, x2, y2, _ = self.boxes[-1]
            if abs(x2-x1) < 5 or abs(y2-y1) < 5:
                self.boxes.pop()
                self.selected_idx = None
                self.update_display()
        self.is_drawing = False
        self.is_resizing = False

    def on_key(self, event):
        if self.selected_idx is None: return
        if event.key in ['0','1','2','3','4','5']:
            self.boxes[self.selected_idx][4] = int(event.key)
        elif event.key in ['delete', 'backspace']:
            self.boxes.pop(self.selected_idx)
            self.selected_idx = None
        self.update_display()

    def save_labels(self, event):
        if not self.loaded_filename: return
        h, w = self.data.shape[:2]
        txt_path = os.path.splitext(self.loaded_filename)[0] + ".txt"
        with open(txt_path, 'w') as f:
            for x1, y1, x2, y2, cls in self.boxes:
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                xc, yc = (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h
                nw, nh = (xmax - xmin) / w, (ymax - ymin) / h
                f.write(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
        print(f"Labels saved to: {txt_path}")

    def update_brightness(self, val):
        self.brightness = val
        self.update_display()

    def manual_load_label(self, event):
        # Implementation of label loading logic...
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Path to YOLO model")
    parser.add_argument('--path', type=str, help="Initial data directory")
    args = parser.parse_args()
    
    app = MSILabelerRGB(model_path=args.model, init_path=args.path)
    plt.show()