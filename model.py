import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

data = pd.read_csv(".\\Salaries.csv")

# Years since PhD and Years in Service
X = data.iloc[:,[3,4]]

# Salary of Professor
y = data.iloc[:,6]

# Linear Regression model
model = LinearRegression()
model.fit(X,y)

# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))





