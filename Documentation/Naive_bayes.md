# Naive Bayes: Concise Documentation

## Definition and Purpose
- **Naive Bayes** is a family of supervised machine learning algorithms used primarily for **classification** tasks.
- It is based on **Bayes’ theorem** and assumes that all features are *conditionally independent* given the class label (the "naive" assumption).

## Core Concepts

- **Bayes’ Theorem**:  
  \[
  P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}
  \]
  Where:
  - \( P(y|X) \): Posterior probability of class \( y \) given features \( X \)
  - \( P(X|y) \): Likelihood of features \( X \) given class \( y \)
  - \( P(y) \): Prior probability of class \( y \)
  - \( P(X) \): Marginal probability of features \( X \)[1][6]

- **Naive Assumption**:  
  All features are independent given the class.  
  \[
  P(X|y) = \prod_{i=1}^{n} P(x_i|y)
  \]

- **Classification Rule**:  
  Assign the class with the highest posterior probability (Maximum A Posteriori, MAP estimate).

## Types of Naive Bayes Classifiers

| Type                  | Description                                              | Typical Use Case           |
|-----------------------|---------------------------------------------------------|----------------------------|
| Gaussian Naive Bayes  | Assumes features are normally distributed               | Continuous features        |
| Multinomial Naive Bayes| For count data (e.g., word counts in text)             | Text classification        |
| Bernoulli Naive Bayes | For binary/boolean features                             | Binary feature data        |

## Algorithm Steps

1. **Estimate Priors**: Compute \( P(y) \) for each class from the training data.
2. **Estimate Likelihoods**: Compute \( P(x_i|y) \) for each feature and class.
3. **Prediction**: For a new instance, calculate the posterior for each class and assign the class with the highest posterior.

## Key Parameters

| Parameter      | Description                                                     |
|----------------|-----------------------------------------------------------------|
| alpha (α)      | Smoothing parameter to avoid zero probabilities (Laplace/Lidstone smoothing) |
| binarize       | Threshold for converting features to binary (Bernoulli NB)   |
| fit_prior      | Whether to learn class priors or use uniform priors             |

## Practical Tips

- **Data Requirements**: Works best when features are independent and relevant to the class.
- **Speed**: Extremely fast to train and predict, suitable for large datasets[1][9].
- **Probability Estimates**: Naive Bayes is a good classifier but not a good probability estimator; predicted probabilities may not be well-calibrated.
- **Handling Zero Counts**: Use smoothing (e.g., alpha > 0) to avoid zero probabilities for unseen feature-class combinations[3].
- **Feature Scaling**: Not required for Naive Bayes, except for Gaussian NB, where normalization may help.

## Implementation Example (scikit-learn)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
## Use Cases

- **Text classification** (spam detection, sentiment analysis)
- **Document categorization**
- **Medical diagnosis**
- **Recommendation systems**

## Advantages

- **Simple and easy to implement**
- **Fast training and prediction**
- **Performs well with high-dimensional data**
- **Works well with small datasets**

## Limitations

- **Assumes feature independence** (rarely true in real data)
- **Poor probability estimates**
- **Not suitable for highly correlated features**

## Summary Table: Key Naive Bayes Parameters

| Parameter   | Description                                                      |
|-------------|------------------------------------------------------------------|
| alpha       | Smoothing parameter to handle zero counts (default: 1.0)         |
| binarize    | Threshold for binarizing features (Bernoulli NB)                 |
| fit_prior   | Whether to learn class prior probabilities from data             |
| class_prior | Prior probabilities of the classes (optional override)           |

---


