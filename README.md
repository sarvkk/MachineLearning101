# Simple and Multiple Linear Regression Implementation

A lightweight implementation of **Linear Regression** from scratch using NumPy, along with a **Multiple Linear Regression** model using Scikit-Learn. This project provides an educational approach to understanding the fundamentals of regression, gradient descent, and evaluation metrics.

## Features

- Custom implementation of **Simple Linear Regression** using gradient descent
- Multiple Linear Regression implementation using **Scikit-Learn**
- Configurable learning rate and convergence criteria
- R-squared (R²) score calculation
- Visualization tools for model fitting and prediction
- Convergence tracking
- **Pure NumPy implementation** (for Simple Linear Regression, without Scikit-Learn dependency)
- **Scikit-Learn-based implementation** for Multiple Linear Regression

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

## Parameters

- **Simple Linear Regression**
  - `lr` (float): Learning rate for gradient descent (default: 0.01)
  - `max_iter` (int): Maximum number of iterations for gradient descent (default: 2000)
  - `threshold` (float): Convergence threshold for loss change (default: 1e-6)

## Methods

- `fit(X, y)`: Train the model on input data
- `predict(X)`: Make predictions on new data
- `r2_score(y_true, y_pred)`: Calculate R-squared score
- `plot(X, y)`: Visualize the regression line and data points

## Example Output

The `plot` method generates a visualization showing:
- Original data points (blue scatter points)
- Fitted regression line (red)
- R-squared score
- Labeled axes and title
- Grid for better readability

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- This implementation is meant for educational purposes
- Inspired by the need for a simple, understandable regression implementation

