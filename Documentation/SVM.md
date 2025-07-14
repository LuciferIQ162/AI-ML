# Support Vector Machine (SVM): Concise Documentation

## Definition and Purpose
- **SVM** is a supervised machine learning algorithm used for **classification**, **regression**, and **outlier detection**. It is most commonly used for classification tasks.

## Core Concepts
- **Hyperplane**: The decision boundary that separates classes in the feature space. For linear classification, it is defined by:  
  $$ w^T x + b = 0 $$
- **Support Vectors**: The data points closest to the hyperplane; they are critical for defining the margin and the position of the hyperplane.
- **Margin**: The distance between the hyperplane and the nearest support vectors from either class. SVM maximizes this margin for better generalization.

## Algorithm Overview
- For **linearly separable data**, SVM finds the hyperplane with the maximum margin.
- For **non-linear data**, SVM uses a **kernel function** to map data into a higher-dimensional space where a linear separator (hyperplane) can be found.

## Kernels
- **Linear**: Suitable for linearly separable data.
- **Polynomial**: For polynomial relationships.
- **Radial Basis Function (RBF)**: Handles non-linear separation by considering the distance between points.
- **Sigmoid**: Similar to neural networks.

You can also define **custom kernels** by providing a function or a precomputed Gram matrix.

## Mathematical Formulation (Linear SVM)
$$
\min_{\mathbf{w},b} \frac{1}{2}|\mathbf{w}|^2 \quad \text{subject to} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \quad i=1,\ldots,n
$$
This is a convex quadratic programming problem.

## Soft Margin and Regularization
- **Soft Margin**: Allows some misclassification to handle non-separable data using slack variables.
- **C (Regularization parameter)**: Balances margin maximization and misclassification penalty. Lower C increases regularization (wider margin, more tolerance for misclassification); higher C reduces tolerance for misclassification.

## Practical Tips
- **Data Scaling**: SVMs are not scale-invariant. Always scale or standardize your data for optimal performance.
- **Kernel Cache Size**: Increasing `cache_size` can speed up training for large datasets.
- **Randomness**: Some SVM implementations use randomness (e.g., in probability estimation); control with `random_state` if reproducibility is needed.
- **Class Imbalance**: Use `class_weight='balanced'` for unbalanced datasets.

## Implementation Example (scikit-learn)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale'))
clf.fit(X_train, y_train)




## Use Cases
- Binary and multiclass classification (e.g., spam detection, image recognition)
- Regression (SVR)
- Outlier/novelty detection (One-Class SVM)

## Summary Table: Key SVM Parameters

| Parameter     | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| kernel        | Type of kernel function (linear, poly, rbf, sigmoid, custom)                |
| C             | Regularization parameter, controls trade-off between margin and misclassification |
| gamma         | Kernel coefficient for RBF, poly, sigmoid kernels (influence of single training examples) |
| degree        | Degree of the polynomial kernel function                                    |
| coef0         | Independent term in poly and sigmoid kernels                                |
| class_weight  | Adjusts weights for classes (useful for imbalanced data)                    |
