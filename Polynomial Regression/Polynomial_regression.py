import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values


from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_poly,y)

#Plot the regression line
plt.scatter(X,y,c='r')
plt.plot(X,regressor.predict(X_poly))
plt.xlabel("Salary")
plt.ylabel("Lavel")
plt.show()
