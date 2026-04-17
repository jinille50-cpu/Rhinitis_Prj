import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tifffile 
from matplotlib.widgets import Slider, Button, TextBox, RadioButtons, CheckButtons
from tkinter import filedialog
import tkinter as tk
import h5py 
import os
from PIL import Image
import tifffile as tiff 


class SpectrumViewer:
    def __init__(self):
        self.data = None
        self.points = []
        self.colors = ["Red", '#1f77b4', '#ff7f0e', "#2ca02c", "#b6d627", '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
        self.bin_size = 1
        self.spectrum_cursor = None  
        self.cursor_band = 0  
        self.markers = []
        self.rectangles = []
        self.rgb_bands = [100,330,450]
        self.initialdir=os.getcwd()  
        self.apply_snv = False  
        self.autoscale = True 
        self.loaded_filename = True 
        self.export_filename = 'Name' 
        self.rgb_cursors = [None, None, None]  
        self.dragging_cursor = None  
        
        self.fig = plt.figure(figsize=(16, 9))
        gs = self.fig.add_gridspec(3, 3, height_ratios=[3, 2, 0.3], width_ratios=[2, 2, 1])
        
  
        self.ax_img = self.fig.add_subplot(gs[0, 0])
        self.ax_img.set_title('No image loaded - Click "Load Image"')
        self.ax_img.axis('off')
        self.im = None
        

        self.filename_text = self.ax_img.text(2.5, -0.05, '', 
                                             transform=self.ax_img.transAxes,
                                             ha='center', va='top',
                                             fontsize=10, style='italic',
                                             bbox=dict(boxstyle='round,pad=0.5', 
                                                      facecolor='wheat', alpha=0.8))
        
    
        self.ax_spec = self.fig.add_subplot(gs[0, 1])
        self.spec_lines = []
        self.ax_spec.set_xlabel('Band Number')
        self.ax_spec.set_ylabel('Intensity')
        self.ax_spec.set_title('Multi-point Spectra')
        self.ax_spec.grid(True)
        
        
        self.ax_band = self.fig.add_subplot(gs[0, 2])
        self.ax_band.set_title('No image loaded')
        self.ax_band.axis('off')
        self.band_img = None
        
        
        self.ax_info = self.fig.add_subplot(gs[1, 0])
        self.ax_info.axis('off')
        self.point_text = self.ax_info.text(0.05, 0.95, '', fontsize=10, verticalalignment='top',
                                           family='monospace')
        self.update_point_list()
        
      
        self.ax_empty = self.fig.add_subplot(gs[1, 1])
        self.ax_empty.axis('off')
        
        self.ax_control = self.fig.add_subplot(gs[1, 2])   
        self.ax_control.axis('off')
        
  
        ax_load = plt.axes([0.8, 0.35, 0.15, 0.05])
        self.btn_load = Button(ax_load, 'Load Image')
        self.btn_load.on_clicked(self.load_image)

    
        self.fig.text(0.42, 0.25, 'False Color (R,G,B):', fontsize=14)
        
        ax_r = plt.axes([0.42, 0.2, 0.04, 0.03])
        ax_g = plt.axes([0.47, 0.2, 0.04, 0.03])
        ax_b = plt.axes([0.52, 0.2, 0.04, 0.03])
        
        self.tb_r = TextBox(ax_r, '', initial=str(self.rgb_bands[0]))
        self.tb_g = TextBox(ax_g, '', initial=str(self.rgb_bands[1]))
        self.tb_b = TextBox(ax_b, '', initial=str(self.rgb_bands[2]))
        
        self.tb_r.on_submit(lambda text: self.update_rgb_bands(0, text))
        self.tb_g.on_submit(lambda text: self.update_rgb_bands(1, text))
        self.tb_b.on_submit(lambda text: self.update_rgb_bands(2, text))
        
      
        ax_snv = plt.axes([0.6, 0.35, 0.15, 0.05])
        self.check_snv = CheckButtons(ax_snv, ['SNV Normalization'], [False])
        self.check_snv.on_clicked(self.toggle_snv)
        
        
        ax_autoscale = plt.axes([0.6, 0.3, 0.15, 0.05])
        self.check_autoscale = CheckButtons(ax_autoscale, ['Autoscale Y-axis'], [True])
        self.check_autoscale.on_clicked(self.toggle_autoscale)
        
        
        ax_bin = plt.axes([0.8, 0.25, 0.15, 0.04])
        self.bin_textbox = TextBox(ax_bin, 'Bin Size:', initial='1')
        self.bin_textbox.on_submit(self.update_bin_size)
     
        ax_clear = plt.axes([0.6, 0.25, 0.15, 0.05])
        self.btn_clear = Button(ax_clear, 'Clear All')
        self.btn_clear.on_clicked(self.clear_all)
        
     
        ax_filename = plt.axes([0.8, 0.15, 0.15, 0.04])
        self.tb_filename = TextBox(ax_filename, 'Filename:', initial=self.export_filename)
        self.tb_filename.on_submit(self.update_filename)
        

        ax_export = plt.axes([0.6, 0.15, 0.15, 0.05])
        self.btn_export = Button(ax_export, 'Export Spectra')
        self.btn_export.on_clicked(self.export_spectra)
        
    
        ax_slider = plt.axes([0.1, 0.08, 0.8, 0.03])
        self.slider = Slider(ax_slider, 'Band', 0, 1, valinit=0, valstep=1)
        self.slider.on_changed(self.update_band)
        self.slider.set_active(False)
        
        
        self.dragging = False
        self.drag_point_idx = None
        
     
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        ax_save_falsecolor = plt.axes([0.42, 0.15, 0.14, 0.04])
        self.btn_save_falsecolor = Button(ax_save_falsecolor, 'Save False Color')
        self.btn_save_falsecolor.on_clicked(self.save_falsecolor_image)

                
    def update_rgb_cursors(self): 
    
        if self.data is None:
            return
            
        colors = ['red', 'green', 'blue']
        labels = ['R', 'G', 'B']
       
        for cursor in self.rgb_cursors:
            if cursor:
                cursor.remove()
        
 
        for i in range(3):
            band = self.rgb_bands[i]
            self.rgb_cursors[i] = self.ax_spec.axvline(
                band, color=colors[i], linestyle='-', linewidth=2, 
                alpha=0.8, label=f'{labels[i]}: {band}'
            )

    def on_motion(self, event):
        
  
        if self.dragging and self.drag_point_idx is not None:
            if event.inaxes == self.ax_img and event.xdata and event.ydata:
                x, y = int(event.xdata), int(event.ydata)
                
                if 0 <= x < self.data.shape[1] and 0 <= y < self.data.shape[0]:
                    
                    old_x, old_y, label = self.points[self.drag_point_idx]
                    self.points[self.drag_point_idx] = (x, y, label)
                    
                   
                    self.plot_all_spectra()
                    self.update_point_list()
        
     
        elif self.dragging_cursor is not None:
            if event.inaxes == self.ax_spec and event.xdata is not None:
                band = int(round(event.xdata))
                if 0 <= band < self.data.shape[2]:
                    
                    self.rgb_bands[self.dragging_cursor] = band
                    
                    if self.dragging_cursor == 0:
                        self.tb_r.set_val(str(band))
                    elif self.dragging_cursor == 1:
                        self.tb_g.set_val(str(band))
                    else:
                        self.tb_b.set_val(str(band))
                    
                    if not hasattr(event, 'button') or event.button is None:
                        if self.dragging_cursor is not None:
                            print("Ending cursor drag via motion")
                            self.dragging_cursor = None
                        if self.dragging:
                            self.dragging = False
                            self.drag_point_idx = None
                        return

                   
                    self.update_rgb_cursors()
                    self.update_rgb_image()

    def on_release(self, event):
       
        if self.dragging:
            self.dragging = False
            self.drag_point_idx = None
    
        if self.dragging_cursor is not None:
            self.dragging_cursor = None
            print("RGB cursor drag ended")



    def update_filename_display(self):
        if self.loaded_filename:
            self.filename_text.set_text(f'File: {self.loaded_filename}')
            self.filename_text.set_visible(True)
        else:
            self.filename_text.set_visible(False)
        self.fig.canvas.draw_idle()
    
    def apply_snv_normalization(self, spectrum):
    
        mean = np.mean(spectrum)
        std = np.std(spectrum)
        if std == 0:
            return spectrum - mean
        return (spectrum - mean) / std
    
    def toggle_snv(self, label):
 
        self.apply_snv = not self.apply_snv
        print(f"SNV Normalization: {'ON' if self.apply_snv else 'OFF'}")
        if self.points:
            self.plot_all_spectra()
    
    def toggle_autoscale(self, label):
     
        self.autoscale = not self.autoscale
        print(f"Autoscale Y-axis: {'ON' if self.autoscale else 'OFF'}")
        if self.points:
            self.plot_all_spectra()
    
    def load_image(self, event):
   
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            data_files = filedialog.askopenfilename(
                title='Select Hyperspectral Image',
                initialdir=self.initialdir,
                filetypes=[
                    ('Hyperspectral files', '*.mat *.h5 *.hdf5 *.tiff *.tif'),
                    ('MAT files', '*.mat'),
                    ('HDF5 files', '*.h5 *.hdf5'),
                    ('TIFF files', '*.tiff *.tif'),
                    ('All files', '*.*')
                ]
            )
            
            root.destroy()
            
            if not data_files:
                print("No file selected")
                return
            
            self.loaded_filename = os.path.basename(data_files)
            base_name = os.path.splitext(self.loaded_filename)[0]
            self.export_filename = base_name
            self.tb_filename.set_val(base_name)
            
            file_ext = data_files.lower().split('.')[-1]
            
            if file_ext == 'mat':
                mat_data = sio.loadmat(data_files)

                if 'image' in mat_data:
                    self.data = mat_data['image']
                    print(f"Using variable: 'image'")
                else:

                    valid_keys = [key for key in mat_data.keys() if not key.startswith('__')]
                    if valid_keys:

                        for key in valid_keys:
                            data_candidate = mat_data[key]
                            if isinstance(data_candidate, np.ndarray) and len(data_candidate.shape) == 3:
                                self.data = data_candidate
                                print(f"Using variable: '{key}' (shape: {self.data.shape})")
                                break
                        

                        if self.data is None:
                            key = valid_keys[0]
                            self.data = mat_data[key]
                            print(f"Using variable: '{key}' (shape: {self.data.shape})")
                    else:
                        raise ValueError("No valid variables found in MAT file")
            
            elif file_ext in ['h5', 'hdf5']:
                with h5py.File(data_files, 'r') as f:

                    if 'image' in f:
                        self.data = f['image'][:]
                        print(f"Using variable: 'image'")
                    else:
                
                        valid_keys = [key for key in f.keys() if isinstance(f[key], h5py.Dataset)]
                        if valid_keys:
                            for key in valid_keys:
                                data_candidate = f[key]
                                if len(data_candidate.shape) == 3:
                                    self.data = data_candidate[:]
                                    print(f"Using variable: '{key}' (shape: {data_candidate.shape})")
                                    break
                
                            if self.data is None:
                                key = valid_keys[0]
                                self.data = f[key][:]
                                print(f"Using variable: '{key}' (shape: {f[key].shape})")
                        else:
                            raise ValueError("No valid datasets found in HDF5 file")

                self.data = np.transpose(self.data, (2, 1, 0))
                print(f"H5 data transposed to: {self.data.shape}")
            
            elif file_ext in ['tiff', 'tif']:
        
                try:
        
                    self.data = tiff.imread(data_files)
                    print(f"TIFF loaded with shape: {self.data.shape}")
                    
           
                    if len(self.data.shape) == 3:
                       
                        if self.data.shape[0] < self.data.shape[2]:
                            self.data = np.transpose(self.data, (1, 2, 0))
                            print(f"TIFF data transposed to: {self.data.shape}")
                    elif len(self.data.shape) == 2:
            
                        self.data = np.expand_dims(self.data, axis=2)
                        print(f"2D TIFF expanded to: {self.data.shape}")
                    else:
                        raise ValueError(f"Unsupported TIFF dimensions: {self.data.shape}")
                        
                except ImportError:
                    print("tifffile library not found. Installing...")
                    try:
                        import subprocess
                        subprocess.check_call(['pip', 'install', 'tifffile'])
                        import tifffile
                        self.data = tifffile.imread(data_files)
                 
                        if len(self.data.shape) == 3 and self.data.shape[0] < self.data.shape[2]:
                            self.data = np.transpose(self.data, (1, 2, 0))
                        elif len(self.data.shape) == 2:
                            self.data = np.expand_dims(self.data, axis=2)
                    except Exception as install_error:
                        print(f"Failed to install tifffile: {install_error}")
                        return
            else:
                print(f"Unsupported file format: .{file_ext}")
                return
            
            self.data = self.data.astype(np.float32)
            
            print(f"Data shape: {self.data.shape}")
            print(f"Data type: {self.data.dtype}")
            print(f"Data range: [{self.data.min()}, {self.data.max()}]")
            
            self.points = []
            self.clear_markers()
            self.clear_spectra()
            
            max_band = self.data.shape[2] - 1
            self.rgb_bands = [min(b, max_band) for b in self.rgb_bands]
            
            self.update_rgb_image()
            
            self.ax_band.clear()
            self.ax_band.axis('on')
            self.band_img = self.ax_band.imshow(self.data[:, :, 0], cmap='gray')
            self.ax_band.set_title('Band 0')
            
            self.slider.valmin = 0
            self.slider.valmax = self.data.shape[2] - 1
            self.slider.set_val(0)
            self.slider.ax.set_xlim(0, self.data.shape[2] - 1)
            self.slider.set_active(True)
            
            self.update_point_list()
            self.update_filename_display() 
            self.fig.canvas.draw_idle()
            print("Image loaded successfully!")
            print(f"Export filename set to: {self.export_filename}")
            
        except Exception as e:
            print(f"Error loading image: {e}")
            import traceback
            traceback.print_exc()
    
    def update_rgb_bands(self, idx, text):
        if self.data is None:
            print("Please load an image first")
            return
        
        try:
            band = int(text)
            if 0 <= band < self.data.shape[2]:
                self.rgb_bands[idx] = band
                self.update_rgb_image()
            else:
                print(f"Band number must be between 0 and {self.data.shape[2]-1}")
                if idx == 0:
                    self.tb_r.set_val(str(self.rgb_bands[0]))
                elif idx == 1:
                    self.tb_g.set_val(str(self.rgb_bands[1]))
                else:
                    self.tb_b.set_val(str(self.rgb_bands[2]))
        except ValueError:
            print("Invalid band number")
    
    def save_falsecolor_image(self, event):
        if self.data is None:
            print("No image loaded to save")
            return
        
        try:

            rgb = self.data[:, :, self.rgb_bands].astype(np.float32)

            rgb_norm = np.zeros_like(rgb)
            for i in range(3): 
                channel = rgb[:, :, i]
                low_val = np.percentile(channel, 0.1)
                high_val = np.percentile(channel, 99.5)

                channel_clipped = np.clip(channel, low_val, high_val)
                rgb_norm[:, :, i] =( (channel_clipped - low_val) / (high_val - low_val + 1e-8))*1.2


            rgb_norm = np.clip(rgb_norm, 0, 1)  
            rgb_norm = np.nan_to_num(rgb_norm)  
            rgb_image = (rgb_norm * 255).astype(np.uint8)

            base_name = self.export_filename if self.export_filename != 'Name' else 'falsecolor'
            r_band, g_band, b_band = self.rgb_bands
            filename = f'{base_name}_falsecolor_R{r_band}_G{g_band}_B{b_band}.png'
            
    
            pil_image = Image.fromarray(rgb_image)
            pil_image.save(filename)
            
            print(f"False color image saved as: {filename}")
            print(f"RGB bands used: R={r_band}, G={g_band}, B={b_band}")

            print(f"Channel normalization ranges:")
            for i, channel_name in enumerate(['R', 'G', 'B']):
                channel = rgb[:, :, i]
                low_val = np.percentile(channel, 0.1)
                high_val = np.percentile(channel, 99.5)
                print(f"  {channel_name}-channel: [{low_val:.4f}, {high_val:.4f}]")
            
        except Exception as e:
            print(f"Error saving false color image: {str(e)}")

    def update_rgb_image(self):
        if self.data is None:
            return
    
        self.ax_img.clear()
        self.markers = []    
        self.rectangles = []      
        self.ax_img.axis('on')

        rgb = self.data[:, :, self.rgb_bands].astype(np.float32)

   
        rgb_norm = np.zeros_like(rgb)
        for i in range(3):  
            channel = rgb[:, :, i]
            low_val = np.percentile(channel, 0.1)
            high_val = np.percentile(channel, 99.9)
       
            channel_clipped = np.clip(channel, low_val, high_val)
            rgb_norm[:, :, i] = ((channel_clipped - low_val) / (high_val - low_val + 1e-8))*1.2
        
        self.im = self.ax_img.imshow(rgb_norm)
        self.ax_img.set_title(f'File : {self.loaded_filename}' if self.loaded_filename else 'No image loaded')

        self.plot_all_spectra()

    def get_binned_spectrum(self, x, y):
  
        if self.data is None:
            return None, None
            
        half_bin = self.bin_size // 2
        y_start = max(0, y - half_bin)
        y_end = min(self.data.shape[0], y + half_bin + 1)
        x_start = max(0, x - half_bin)
        x_end = min(self.data.shape[1], x + half_bin + 1)
        
        region = self.data[y_start:y_end, x_start:x_end, :]
        spectrum = np.mean(region, axis=(0, 1))
    
        if self.apply_snv:
            spectrum = self.apply_snv_normalization(spectrum)
        
        return spectrum, (y_start, y_end, x_start, x_end)
    
    def onclick(self, event):
        if self.data is None:
            print("Please load an image first")
            return
        

        if event.inaxes == self.ax_img and event.xdata and event.ydata:
            x, y = int(event.xdata), int(event.ydata)
            
            if 0 <= x < self.data.shape[1] and 0 <= y < self.data.shape[0]:
                if event.button == 1:  
     
                    if self.points:
                        distances = [(i, np.sqrt((x - px)**2 + (y - py)**2)) 
                                for i, (px, py, _) in enumerate(self.points)]
                        nearest_idx, nearest_dist = min(distances, key=lambda item: item[1])
                        
                
                        if nearest_dist < 5:
                            self.dragging = True
                            self.drag_point_idx = nearest_idx
                            return

                    label = f'P{len(self.points) + 1}'
                    self.points.append((x, y, label))
                    self.plot_all_spectra()
                    self.update_point_list()
                    
                elif event.button == 3: 
                    if self.points:
                        distances = [(i, np.sqrt((x - px)**2 + (y - py)**2)) 
                                for i, (px, py, _) in enumerate(self.points)]
                        nearest_idx = min(distances, key=lambda item: item[1])[0]
                        self.points.pop(nearest_idx)
                        self.plot_all_spectra()
                        self.update_point_list()
        

        elif event.inaxes == self.ax_spec and event.xdata is not None:
            band = event.xdata
   
            distances = [abs(band - self.rgb_bands[i]) for i in range(3)]
            nearest_idx = distances.index(min(distances))
            
  
            if distances[nearest_idx] < 5:
                self.dragging_cursor = nearest_idx
                print(f"Started dragging {['R', 'G', 'B'][nearest_idx]} cursor")
    


    def on_scroll(self, event):
        if event.inaxes == self.ax_img and self.data is not None:
            if event.xdata is None or event.ydata is None:
                return
                
            zoom_factor = 1.2 if event.button == 'up' else 1/1.2
            xlim = self.ax_img.get_xlim()
            ylim = self.ax_img.get_ylim()
            img_height, img_width = self.data.shape[:2]

            width = xlim[1] - xlim[0]
            height = ylim[0] - ylim[1]  
            

            new_width = width / zoom_factor
            new_height = height / zoom_factor
            

            min_width = 10  
            min_height = 10  
            new_width = max(new_width, min_width)
            new_height = max(new_height, min_height)
            
            new_width = min(new_width, img_width)
            new_height = min(new_height, img_height)
            
            center_x, center_y = event.xdata, event.ydata

            new_x0 = center_x - new_width / 2
            new_x1 = center_x + new_width / 2
            new_y0 = center_y + new_height / 2  
            new_y1 = center_y - new_height / 2  
            
            if new_x0 < 0:
                shift = -new_x0
                new_x0 = 0
                new_x1 = min(img_width, new_x1 + shift)
            elif new_x1 > img_width:
                shift = new_x1 - img_width
                new_x1 = img_width
                new_x0 = max(0, new_x0 - shift)
                
            if new_y1 < 0:
                shift = -new_y1
                new_y1 = 0
                new_y0 = min(img_height, new_y0 + shift)
            elif new_y0 > img_height:
                shift = new_y0 - img_height
                new_y0 = img_height
                new_y1 = max(0, new_y1 - shift)
            
            self.ax_img.set_xlim(new_x0, new_x1)
            self.ax_img.set_ylim(new_y0, new_y1)  
            
            self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.dragging:
            self.dragging = False
            self.drag_point_idx = None


    
    def clear_markers(self):
        for marker in self.markers:
            marker.remove()
        for rect in self.rectangles:
            rect.remove()
        self.markers = []
        self.rectangles = []
    
    def clear_spectra(self):
        for line in self.spec_lines:
            line.remove()
        self.spec_lines = []
        if self.ax_spec.get_legend():
            self.ax_spec.get_legend().remove()
    
    def plot_all_spectra(self):
        if self.data is None:
            return
            
        self.clear_spectra()
        self.clear_markers()
        
        self.ax_spec.set_xlabel('Band Number')
        self.ax_spec.set_ylabel('Intensity' + (' (SNV Normalized)' if self.apply_snv else ''))
        title = f'Multi-point Spectra (Bin: {self.bin_size}x{self.bin_size})'
        if self.apply_snv:
            title += ' - SNV ON'
        self.ax_spec.set_title(title)
        self.ax_spec.grid(True)
        
        for i, (x, y, label) in enumerate(self.points):
            color = self.colors[i % len(self.colors)]
            spectrum, region_info = self.get_binned_spectrum(x, y)
            
            if spectrum is None:
                continue
            
            y_start, y_end, x_start, x_end = region_info
            
            line, = self.ax_spec.plot(range(len(spectrum)), spectrum, 
                            color=color, label=f'{label} ({x},{y})', linewidth=2)
            self.spec_lines.append(line)
            
            marker, = self.ax_img.plot(x, y, '+', color=color, markersize=15, markeredgewidth=2)
            self.markers.append(marker)
            
            if self.bin_size > 1:
                rect = plt.Rectangle((x_start-0.5, y_start-0.5), 
                                    x_end-x_start, y_end-y_start,
                                    fill=False, edgecolor=color, linewidth=1.5, linestyle='--')
                self.ax_img.add_patch(rect)
                self.rectangles.append(rect)
        
        if self.points:
            self.ax_spec.legend(loc='best', fontsize=8)
            

            if self.autoscale:
                self.ax_spec.relim()
                self.ax_spec.autoscale_view()
            else:

                pass
        if self.points:  
            self.update_rgb_cursors()
        
        self.fig.canvas.draw_idle()

    
    def update_point_list(self):
        if self.data is None:
            text = "No image loaded\n\n"
            text += "Click 'Load Image' to start"
        elif self.points:
            text = "Selected Points:\n" + "="*30 + "\n"
            for i, (x, y, label) in enumerate(self.points):
                text += f"{label}: (x={x:3d}, y={y:3d})\n"
            if self.apply_snv:
                text += "\n[SNV: ON]"
        else:
            text = "No points selected\n\n"
            text += "Instructions:\n"
            text += "- Left click: Add point\n"
            text += "- Drag point: Move point\n"

            text += "- Right click: Remove nearest"
        
        self.point_text.set_text(text)
        self.fig.canvas.draw_idle()
    
    def update_band(self, val):
        if self.data is None:
            return
        band = int(self.slider.val)
        band_data = self.data[:, :, band].astype(np.float32)
        self.band_img.set_data(band_data)
        self.band_img.set_clim(band_data.min(), band_data.max())
        self.ax_band.set_title(f'Band {band}')
        self.fig.canvas.draw_idle()
    
    def update_bin_size(self, text):
        try:
            new_size = int(text)
            if new_size > 0 and new_size < 50:
                self.bin_size = new_size
                if self.points:
                    self.plot_all_spectra()
            else:
                print("Bin size must be between 1 and 50")
                self.bin_textbox.set_val(str(self.bin_size))
        except ValueError:
            print("Invalid bin size")
            self.bin_textbox.set_val(str(self.bin_size))
    
    def update_filename(self, text):
        if text.endswith('.txt'):
            text = text[:-4]
        self.export_filename = text
        print(f"Export filename set to: {self.export_filename}.txt")
    
    def clear_all(self, event):
        self.points = []
        self.plot_all_spectra()
        self.update_point_list()
    
    def export_spectra(self, event):
        if not self.points:
            print("No points to export")
            return
        
        all_spectra = []
        
        for x, y, label in self.points:
            spectrum, _ = self.get_binned_spectrum(x, y)
            all_spectra.append(spectrum)
        
        spectra_array = np.column_stack(all_spectra)
        band_numbers = np.arange(len(all_spectra[0])).reshape(-1, 1)
        
        output_data = np.hstack([band_numbers, spectra_array])
        
        filename = f'{self.export_filename}.txt'
        if self.apply_snv:
            filename = f'{self.export_filename}_SNV.txt'
        
        np.savetxt(filename, output_data, fmt='%.6f', delimiter='\t')
        
        print(f"Spectra exported to {filename}")
        if self.apply_snv:
            print("(SNV normalization applied)")
        print(f"Format: Band_Number | {' | '.join([label for _, _, label in self.points])}")
    
    def show(self):
        plt.tight_layout()
        plt.show()


viewer = SpectrumViewer()
viewer.show()