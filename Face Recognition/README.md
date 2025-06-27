# Face Recognition (MATLAB)

This folder features scripts for face recognition using linear algebra approaches—PCA (Principal Component Analysis) and SVD (Singular Value Decomposition).

## Files

- **Traindb.m**  
  Trains a face recognition model using PCA on your training face dataset.
- **Testdb.m**  
  Tests the PCA-trained model on your test set, reporting recognition accuracy.
- **Trainnsvd.m**  
  Trains a face recognition model using SVD.
- **Testsvd.m**  
  Tests the SVD-trained model on your test images.

## How to Use

1. Place your training and test face images as required by the scripts.
2. Open MATLAB and set your working directory to `face_recognition`.
3. To perform PCA-based recognition:
   ```matlab
   Traindb    % Train PCA model
   Testdb     % Test PCA model
   ```
4. For SVD-based recognition:
   ```matlab
   Trainnsvd  % Train SVD model
   Testsvd    % Test SVD model
   ```

## Requirements

- MATLAB (R2019b or newer recommended)
- Image Processing Toolbox (recommended)

---

For questions or feedback, please contact [Nitir4](https://github.com/Nitir4).
