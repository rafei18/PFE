# Multispectral image classification using linear programming and the multicriteria decision-making method "AHP".

This is part of the work that I conducted for my end-of-studies project in 2023
#
The idea is to create a soft voting ensemble learning set, where multiple classifiers cooperate to improve the quality of classification and achieve a balanced compromise between bias and variance (i.e., Under/Overfitting).

## 1. Sanderband dataset with provided ground truth:
    - Building a generalized model for deriving weights from the pairwise comparison matrix using the PuLP library.
    - Loading and reading the data  ".TIFF" using the library "rasterio".
    - Visualization of the different spectral bands. 
    - Preprocessing: Normalization of the data.
    - Model training: In this step, multiple models are trained while performing hyperparameter tuning using "nested cross-validation." Finally, the best parameters are used to construct the ensemble learning using weights calculated previously.

## 2. Aquileria_italy dataset with no ground truth:
    - For this scenario, we utilized unsupervised learning with K-means to generate a ground truth. We employed the elbow method and silhouette score to determine the optimal value of K.