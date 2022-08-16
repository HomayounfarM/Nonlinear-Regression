#Random forest regression

# Importnigg the library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# Importing the dataset

for file in os.listdir():
    if file.endswith('.csv'):
        name = file
dataset = pd.read_csv(name)

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Fitting random forest regression to the data set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
sample = 6.5
sample = np.array(sample).reshape(-1,1)
y_pred = regressor.predict(sample)

# Visualising the regression results (For high resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('True or Bluf - Random Forest')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()





