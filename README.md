Machine Learning Algorithms
===========================

This repository contains implementations for various machine learning algorithms, including **Simple Linear Regression**, **Custom Logistic Regression**, **Decision Tree Classifier**, **Random Forest Classifier**, and **Naive Bayes Classifier**. Each algorithm is implemented in a simple and understandable way, complete with visualizations to help you track model performance.

Algorithms and Parameters
-------------------------

### Simple Linear Regression

*   **Parameters:**
    
    *   lr (float): Learning rate for gradient descent (default: 0.01)
        
    *   max\_iter (int): Maximum number of iterations for gradient descent (default: 2000)
        
    *   threshold (float): Convergence threshold for loss change (default: 1e-6)
        
*   **Methods:**
    
    *   fit(X, y): Train the model on input data.
        
    *   predict(X): Make predictions on new data.
        
    *   r2\_score(y\_true, y\_pred): Calculate the R-squared score.
        
    *   plot(X, y): Visualize the regression line and data points.
        
        *   **Visualization Details:**
            
            *   **Blue Scatter Points:** Original data points.
                
            *   **Red Line:** Fitted regression line.
                
            *   **Metrics:** R-squared score displayed.
                
            *   **Extras:** Labeled axes, title, and grid for enhanced readability.
                

### Custom Logistic Regression

*   **Parameters:**
    
    *   lr (float): Learning rate for gradient descent (default: 0.1)
        
    *   n\_iters (int): Number of iterations for training (default: 1000)
        
*   **Methods:**
    
    *   fit(X, y): Train the model on input data.
        
    *   predict(X): Make predictions on new data.
        
    *   binary\_cross\_entropy(y\_true, y\_pred): Calculate the loss function.
        
*   **Visualization & Evaluation:**
    
    *   plot\_loss\_history(): Plot loss function values over training iterations to track convergence.
        
    *   confusion\_matrix(y\_true, y\_pred): Display a confusion matrix.
        
    *   accuracy(y\_true, y\_pred): Compute the classification accuracy.
        

### Decision Tree Classifier

*   **Parameters:**
    
    *   max\_depth (int): Maximum depth of the tree (default: 100)
        
    *   min\_samples\_split (int): Minimum number of samples required to split a node (default: 2)
        
    *   n\_features (int): Number of features to consider when looking for the best split (default: all features)
        
*   **Methods:**
    
    *   fit(X, y): Train the decision tree using entropy-based splitting.
        
    *   predict(X): Make predictions on test data.
        
    *   entropy(y): Calculate entropy for potential splits.
        
    *   accuracy(y\_true, y\_pred): Compute the classification accuracy.
        
*   **Output:**
    
    *   Displays the trained tree structure.
        
    *   Provides predictions on test data along with an accuracy score.
        

### Random Forest Classifier

_Built on top of the DecisionTree class, this classifier implements an ensemble of decision trees._

*   **Parameters:**
    
    *   n\_trees (int): Number of trees in the forest (default: 10).
        
    *   max\_depth (int): Maximum depth of each tree (default: 10).
        
    *   min\_samples\_split (int): Minimum number of samples required to split a node (default: 2).
        
    *   n\_feature (int or None): Number of features to consider when looking for the best split (default: None).
        
*   **Methods:**
    
    *   fit(X, y): Train the random forest by building multiple decision trees using bootstrap samples.
        
    *   predict(X): Make predictions on new data by aggregating the predictions from each tree.
        
    *   \_bootstrap\_samples(X, y): Create bootstrap samples using random sampling with replacement. _(Internal method)_
        
    *   \_most\_common\_label(y): Determine the most common label among predictions. _(Internal method)_
        
*   **Output:**
    
    *   Provides an accuracy score for the classifier.
        
    *   Visualizes predictions versus true labels using scatter plots.
        

### Naive Bayes Classifier

*   **Parameters:**
    
    *   No hyperparameters are required for this basic implementation.
        
*   **Methods:**
    
    *   fit(X, y): Train the classifier.
        
    *   predict(X): Make predictions on test data.
        
    *   accuracy(y\_true, y\_pred): Compute the classification accuracy.
        
*   **Output:**
    
    *   Provides predictions on test data.
        
    *   Displays the accuracy score for model performance.
        

Contributing
------------

Contributions are welcome! Please feel free to submit a Pull Request with any improvements or bug fixes.

Acknowledgments
---------------

This implementation is intended for educational purposes and is inspired by the need for clear and straightforward regression, classification, decision tree, random forest, and Naive Bayes implementations.

Feel free to modify or extend this documentation as needed. Happy coding!