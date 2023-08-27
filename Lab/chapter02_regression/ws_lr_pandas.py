import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
dataframe = pd.read_csv('c:\pyml_scripts\chapter02_regression\wslw.csv',names=['Long','Width'])
x_values = dataframe[['Long']]
y_values = dataframe[['Width']]
Width_reg = linear_model.LinearRegression()
Width_reg.fit(x_values,y_values)
plt.scatter(x_values,y_values,c='red')
plt.plot(x_values,Width_reg.predict(x_values))
plt.show()