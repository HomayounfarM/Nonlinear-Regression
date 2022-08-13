# Decision tree regression

# Importing the libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# Importing the dataset
for file in os.listdir():
    if file.endswith('.csv'):
        name = file

dataset = pd.read_csv(name)
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Fitting the Decision Tree Regression to the dataset

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new resuls

sample = np.array([[6.5]])

y_pred = regressor.predict(sample)

# Visualising the decission tree regression result (for higher resolution and smoother curve)
x_grid = np.arange(min(X), max(X), 0.01)
x_grid = x_grid.reshape(-1,1)
plt.scatter(X, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

