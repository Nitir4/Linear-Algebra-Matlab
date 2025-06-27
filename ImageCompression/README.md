# Image Compression (MATLAB)

This folder contains scripts for compressing images using linear algebra techniques—specifically, Principal Component Analysis (PCA) and Singular Value Decomposition (SVD).

---

## Files

- **ImageCompressionApp_mech505.m**  
  Interactive, text-based script for image compression using PCA or SVD (no GUI).  
  This code lets you:
    1. Load an image (JPG, PNG, BMP, etc.)
    2. Choose to work in RGB (color) or grayscale mode
    3. Compress the image using either PCA or SVD
    4. Compare both methods and analyze their performance
    5. Visualize results and metrics, and optionally save the output image

- **ImageCompressionGUI.m**  
  GUI implementation of the above app (coded in `.m`, not `.mlapp`).  
  Provides a graphical interface for all the above steps, making it easier to interactively:
    - Load images
    - Pick color or grayscale mode
    - Select PCA or SVD for compression
    - Compare methods and visualize results
    - Save compressed images if desired

---

## How to Use

1. Open MATLAB and set your working directory to `image_compression`.
2. To use the interactive, text-based script, type:
   ```matlab
   ImageCompressionApp_mech505
   ```
3. To use the GUI, type:
   ```matlab
   ImageCompressionGUI
   ```

---

## Requirements

- MATLAB (R2019b or newer recommended)
- Image Processing Toolbox (recommended for handling images)

---

For questions or feedback, please contact [Nitir4](https://github.com/Nitir4).
