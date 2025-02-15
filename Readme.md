# Simple Linear Regression Implementation

A lightweight implementation of Linear Regression from scratch using NumPy. This project provides a simple yet educational approach to understanding the fundamentals of linear regression and gradient descent.

## Features

- Custom implementation of Linear Regression using gradient descent
- Configurable learning rate and convergence criteria
- R-squared (R²) score calculation
- Visualization tools for model fitting and prediction
- Convergence tracking
- Pure NumPy implementation (no scikit-learn dependency)

## Requirements

python
numpy>=1.19.0
matplotlib>=3.3.0

## Installation

Clone this repository:

bash
git clone https://github.com/sarvkk/linear-regression.git
cd linear-regression

## Usage

Here's a simple example of how to use the implementation:

python
from linear_regression import simple_LinReg
import numpy as np

Prepare your data

X = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 4.2, 6.1, 8.3, 9.9])

Create and train the model

model = simple_LinReg(lr=0.01, max_iter=2000, threshold=1e-6)
loss_history = model.fit(X, y)

Make predictions

predictions = model.predict(X)

Visualize results

model.plot(X, y)

## Parameters

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
- Inspired by the need for a simple, understandable linear regression implementation
