# -*- coding: utf-8 -*-
"""Linear regression project using scikit-learn.1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NPETLJfamobLrcUM2lJ-IYs7ra2YnTwR
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Generate 100 random data points for X
y = 4 + 3 * X + np.random.randn(100, 1)  # Generate y with some noise

# Create and fit a linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Make predictions using the trained model
X_new = np.array([[0], [2]])  # Two new data points to predict y values
y_pred = lin_reg.predict(X_new)

# Plot the original data and the linear regression line
plt.scatter(X, y, label='Original Data')
plt.plot(X_new, y_pred, 'r-', label='Linear Regression Line', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Display the linear regression model parameters
print("Intercept (Theta 0):", lin_reg.intercept_[0])
print("Coefficient (Theta 1):", lin_reg.coef_[0][0])