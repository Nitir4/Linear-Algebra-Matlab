# Noise Filtering (MATLAB)

This folder contains scripts for denoising matrices and images using linear algebra techniques—specifically, Principal Component Analysis (PCA) and Singular Value Decomposition (SVD).

---

## Files

- **NoiseFiltering_mech505.m**  
  Interactive, text-based script for noise filtering using PCA or SVD (no GUI).  
  This code lets you:
    1. Load a matrix (CSV, text) or image (JPG, PNG, BMP, etc.)
    2. Choose to work in RGB (color) or grayscale mode
    3. Add noise (impulse or Gaussian) to your data
    4. Denoise the data using either PCA or SVD
    5. Compare both methods and analyze their performance
    6. Visualize results and metrics

- **NoiseFilteringGUI.m**  
  GUI implementation of the above app (coded in `.m`, not `.mlapp`).  
  Provides a graphical interface for all the above steps, making it easier to interactively:
    - Load matrix or image data
    - Pick color or grayscale mode
    - Add impulse or Gaussian noise
    - Denoise using PCA or SVD
    - Compare methods and visualize results

---

## How to Use

1. Open MATLAB and set your working directory to `noise_filtering`.
2. To use the interactive, text-based script, type:
   ```matlab
   NoiseFiltering_mech505
   ```
3. To use the GUI, type:
   ```matlab
   NoiseFilteringGUI
   ```

---

## Requirements

- MATLAB (R2019b or newer recommended)
- Image Processing Toolbox (recommended, especially for image files)

---

For questions or feedback, please contact [Nitir4](https://github.com/Nitir4).
