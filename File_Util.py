import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky


def Uigetfile(initial_dir='C:/Users/User/Desktop'):
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True) 

    file_path = filedialog.askopenfilename( 
        parent=root, 
        initialdir=initial_dir, 
        # filetypes=[("Text files", "*.*")] 
    )
    root.destroy()
    return file_path

def Uigetdir(initial_dir='C:/Users/User/Desktop', title="폴더 선택"):
    root = tk.Tk() 
    root.withdraw()  
    root.attributes('-topmost', True)  

    dir_path = filedialog.askdirectory(  
        parent=root, 
        initialdir=initial_dir, 
        title=title 
    )
    
    root.destroy()
    return dir_path
 
def AsLS(A):
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    
    y = np.array(A)
    m = len(y)
    lambda_val = 10**5
    p = 10**(-4)
    
    e = np.ones(m)
    D = sparse.spdiags([e, -2*e, e], [0, -1, -2], m, m-2).T
    
    w = np.ones(m)
    
    for it in range(5):
        W = sparse.spdiags(w, 0, m, m)
        Z = W + lambda_val * D.T.dot(D)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1-p) * (y < z)
    
    return y - z
def normalize_image(img):
    img = img.astype(np.float64)
    channels = img.shape[2]
    normalized_img = np.zeros_like(img)
    
    for c in range(channels):
        channel = img[:, :, c]
        min_val = np.min(channel)
        max_val = np.max(channel)
        
        if max_val - min_val > 0:
            normalized_img[:, :, c] = (channel - min_val) / (max_val - min_val)
        else:
  
            normalized_img[:, :, c] = channel / 255.0
            
    return (normalized_img * 255).astype(np.uint8)

def SNV(data):
    """
    SNV (Standard Normal Variate) 적용
    각 스펙트럼을 평균 0, 표준편차 1로 정규화
    """

    snv_data = np.zeros_like(data)
    
    for i in range(data.shape[1]):  
        spectrum = data[:, i]
        mean_spectrum = np.mean(spectrum)
        std_spectrum = np.std(spectrum)
        
        if std_spectrum != 0: 
            snv_data[:, i] = (spectrum - mean_spectrum) / std_spectrum
        else:
            snv_data[:, i] = spectrum - mean_spectrum
    
    return snv_data
