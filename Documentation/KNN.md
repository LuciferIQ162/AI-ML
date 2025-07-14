# K-Nearest Neighbors (KNN): Concise Documentation

## Definition and Purpose
- **K-Nearest Neighbors (KNN)** is a simple, non-parametric, supervised machine learning algorithm used for classification and regression tasks.
- It predicts the class (or value) of a new data point based on the majority class (or average value) of its k closest data points in the training set.

## How KNN Works

1. Choose the value of k (number of neighbors to consider).
2. Calculate the distance between the new data point and all points in the training set (commonly using Euclidean distance).
3. Sort the distances and select the k nearest neighbors.
4. For classification: assign the class most common among the k neighbors (majority vote).
5. For regression: assign the average (or weighted average) of the k neighbors' values.

## Key Parameters

| Parameter      | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| n_neighbors    | Number of neighbors to use (k)                                              |
| weights        | Weight function used in prediction ('uniform' or 'distance')                 |
| algorithm      | Algorithm used to compute nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute') |
| metric         | Distance metric for tree ('minkowski', 'euclidean', etc.)                   |

## Choosing the Value of k

- Small k (e.g., 1): sensitive to noise and outliers (high variance, low bias).
- Large k: smoother decision boundaries but may underfit (high bias, low variance).
- Odd k: helps avoid ties in classification.
- Optimal k: often selected using cross-validation or the elbow method.

## Advantages

- Simple to understand and implement.
- No training phase (lazy learner): all computation is deferred until prediction.
- Works well with small to medium-sized datasets.

## Limitations

- Computationally expensive at prediction time for large datasets.
- Sensitive to irrelevant features and the scale of data (feature scaling recommended).
- Does not work well with high-dimensional data (curse of dimensionality).

## Practical Tips

- Feature scaling (normalization/standardization) is important for accurate distance calculations.
- Handle missing values before applying KNN.
- Efficient implementations use data structures like KDTree or BallTree for faster neighbor search.

## Implementation Example (scikit-learn)

# K-Nearest Neighbors (KNN): Concise Documentation

## Definition and Purpose
- **K-Nearest Neighbors (KNN)** is a simple, non-parametric, supervised machine learning algorithm used for classification and regression tasks.
- It predicts the class (or value) of a new data point based on the majority class (or average value) of its k closest data points in the training set.

## How KNN Works

1. Choose the value of k (number of neighbors to consider).
2. Calculate the distance between the new data point and all points in the training set (commonly using Euclidean distance).
3. Sort the distances and select the k nearest neighbors.
4. For classification: assign the class most common among the k neighbors (majority vote).
5. For regression: assign the average (or weighted average) of the k neighbors' values.

## Key Parameters

| Parameter      | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| n_neighbors    | Number of neighbors to use (k)                                              |
| weights        | Weight function used in prediction ('uniform' or 'distance')                 |
| algorithm      | Algorithm used to compute nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute') |
| metric         | Distance metric for tree ('minkowski', 'euclidean', etc.)                   |

## Choosing the Value of k

- Small k (e.g., 1): sensitive to noise and outliers (high variance, low bias).
- Large k: smoother decision boundaries but may underfit (high bias, low variance).
- Odd k: helps avoid ties in classification.
- Optimal k: often selected using cross-validation or the elbow method.

## Advantages

- Simple to understand and implement.
- No training phase (lazy learner): all computation is deferred until prediction.
- Works well with small to medium-sized datasets.

## Limitations

- Computationally expensive at prediction time for large datasets.
- Sensitive to irrelevant features and the scale of data (feature scaling recommended).
- Does not work well with high-dimensional data (curse of dimensionality).

## Practical Tips

- Feature scaling (normalization/standardization) is important for accurate distance calculations.
- Handle missing values before applying KNN.
- Efficient implementations use data structures like KDTree or BallTree for faster neighbor search.

## Implementation Example (scikit-learn)

from sklearn.neighbors import KNeighborsClassifier

Create the model with k=3
knn = KNeighborsClassifier(n_neighbors=3)

Fit the model
knn.fit(X_train, y_train)

Predict the class of new data
y_pred = knn.predict(X_test)


## Use Cases

- Image recognition
- Recommender systems
- Text categorization
- Medical diagnosis

## Summary Table: Key KNN Parameters

| Parameter      | Description                                      |
|----------------|--------------------------------------------------|
| n_neighbors    | Number of neighbors to use (k)                   |
| weights        | 'uniform' or 'distance' weighting of neighbors   |
| algorithm      | Search algorithm: 'auto', 'ball_tree', 'kd_tree', 'brute' |
| metric         | Distance metric: 'euclidean', 'manhattan', etc.  |

---
