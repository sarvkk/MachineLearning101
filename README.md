# Simple, Multiple Linear Regression, Logistic Regression, and Decision Tree Implementation

A lightweight implementation of **Linear Regression**, **Logistic Regression**, and **Decision Trees** from scratch using NumPy, along with **Multiple Linear Regression** and **Logistic Regression** models using Scikit-Learn. This project provides an educational approach to understanding the fundamentals of regression, classification, decision trees, gradient descent, and evaluation metrics.

## Features

- Custom implementation of **Simple Linear Regression** using gradient descent
- **Multiple Linear Regression** implementation using **Scikit-Learn**
- **Logistic Regression** implementation:
  - Custom implementation using NumPy (gradient descent)
  - Scikit-Learn-based implementation
- **Decision Tree Classifier** implementation from scratch using NumPy
- Configurable learning rate and convergence criteria
- R-squared (R²) score calculation (for Linear Regression)
- Binary Cross-Entropy loss tracking (for Logistic Regression)
- **Entropy-based splitting for Decision Trees**
- Visualization tools for model fitting, loss history, confusion matrix, and decision boundaries
- Convergence tracking
- **Pure NumPy implementation** (for Simple Linear Regression, Logistic Regression, and Decision Trees, without Scikit-Learn dependency)
- **Scikit-Learn-based implementation** for Multiple Linear Regression and Logistic Regression

## Requirements

```
python
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
seaborn>=0.11.0
pandas>=1.2.0
```

## Installation

Clone this repository:

```bash
git clone https://github.com/sarvkk/MachineLearning101.git
```

## Usage

### Simple Linear Regression (Custom Implementation)

1. Prepare your dataset
2. Create and train the model
3. Make predictions
4. Visualize results

### Multiple Linear Regression (Using Scikit-Learn)

1. Load dataset
2. Perform feature engineering
3. Split the dataset into training and testing sets
4. Train the model
5. Make predictions
6. Evaluate performance

### Logistic Regression

#### Custom Logistic Regression Implementation

1. Train a logistic regression model using gradient descent
2. Track loss history using **Binary Cross-Entropy Loss**
3. Evaluate performance using **Confusion Matrix** and **Accuracy**
4. Plot the loss function over iterations

#### Scikit-Learn Logistic Regression Implementation

1. Train a logistic regression model using **Scikit-Learn**
2. Make predictions and evaluate model performance
3. Display **Confusion Matrix** using Seaborn

### Decision Tree Classifier (Custom Implementation)

1. Load dataset (e.g., Breast Cancer dataset from Scikit-Learn)
2. Split dataset into training and testing sets
3. Train the **Decision Tree Classifier** on training data
4. Make predictions on test data
5. Evaluate model performance using **accuracy score**

#### Example Usage:

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# Train the Decision Tree Classifier
clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Evaluate accuracy
def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print(f"Decision Tree Accuracy: {acc:.4f}")
```

## Parameters

- **Simple Linear Regression**
  - `lr` (float): Learning rate for gradient descent (default: 0.01)
  - `max_iter` (int): Maximum number of iterations for gradient descent (default: 2000)
  - `threshold` (float): Convergence threshold for loss change (default: 1e-6)

- **Custom Logistic Regression**
  - `lr` (float): Learning rate for gradient descent (default: 0.1)
  - `n_iters` (int): Number of iterations for training (default: 1000)

- **Decision Tree Classifier**
  - `max_depth` (int): Maximum depth of the tree (default: 100)
  - `min_samples_split` (int): Minimum number of samples required to split a node (default: 2)
  - `n_features` (int): Number of features to consider when looking for the best split (default: all features)

## Methods

- `fit(X, y)`: Train the model on input data
- `predict(X)`: Make predictions on new data
- `r2_score(y_true, y_pred)`: Calculate R-squared score (Linear Regression)
- `binary_cross_entropy(y_true, y_pred)`: Calculate loss (Logistic Regression)
- `entropy(y)`: Calculate entropy for Decision Tree splits
- `plot(X, y)`: Visualize the regression line and data points (Linear Regression)
- `plot_loss_history()`: Plot loss function over iterations (Logistic Regression)
- `confusion_matrix(y_true, y_pred)`: Display confusion matrix (Logistic Regression)
- `accuracy(y_true, y_pred)`: Compute classification accuracy

## Example Output

The `plot` method generates a visualization showing:
- Original data points (blue scatter points)
- Fitted regression line (red)
- R-squared score (Linear Regression)
- Labeled axes and title
- Grid for better readability

The `plot_loss_history()` method generates:
- Loss function values over training iterations
- Helps track model convergence

The `Decision Tree` outputs:
- **Trained tree structure** with entropy-based splitting
- **Predictions** on test data
- **Accuracy score** to measure model performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- This implementation is meant for educational purposes
- Inspired by the need for a simple, understandable regression, classification, and decision tree implementation

