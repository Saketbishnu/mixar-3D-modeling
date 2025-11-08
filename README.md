# Mesh Normalization, Quantization & Error Analysis

Author: Saket Bishnu  
Registration Number: RA2211027010059  
Date: November 2025  
Institution: [SRM INSTITUTE OF SCIENCE AND TECHNOLOGY / DSBS]  

---

## Project Overview
This project implements a complete **3D mesh preprocessing pipeline** involving normalization, quantization, reconstruction, and error analysis.  
It is a preparatory step for intelligent 3D mesh systems such as **SeamGPT**, which require clean, normalized, and quantized data for AI training.

The pipeline ensures that all meshes have consistent coordinate ranges and quantized formats, enabling fair learning and accurate geometric analysis.


## Key Concepts

### **1. Normalization**
- **Min–Max Normalization**:  
  Scales each axis independently into the range [0, 1].  
  Formula:  
  \[x' = \frac{x - x_{min}}{x_{max} - x_{min}}\]

- **Unit Sphere Normalization**:  
  Centers the mesh and scales it so all vertices lie inside a sphere of radius 1.  
  Formula:  
  \[x' = \frac{x - \mu}{r}\]

---

### **2. Quantization**
Discretizes continuous coordinates into integer bins.  
For 1024 bins:  
\[q = int(x' \times (1024 - 1))\]

Dequantization reverses the process:  
\[x'' = \frac{q}{1023}\]

### **3. Error Measurement**
To assess reconstruction quality:
**(MSE)**
**(MAE)**  
Computed between:
Original vertices  
Reconstructed (dequantized + denormalized) vertices
## Implementation Details

### **Languages & Libraries**
Python 3.10
NumPy  
Matplotlib  
(Optional) Open3D or Trimesh for visualization

### **Execution Environment**
Anaconda Environment: meshproc

### outputs
## How to Run

1. Activate your Conda environment:
   conda activate meshproc

### Run the main preprocessing script:
python mesh_preprocessing.py --input_dir "data\8samples" --output_dir "outputs" --bins 1024
