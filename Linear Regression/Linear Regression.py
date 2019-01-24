import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv("F:\DB\student_scores.csv")

dataset.isnull().sum()

X=dataset.iloc[:,:1].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(X_train,y_train)


pred=regressor.predict(X_test)


#plot train results
plt.scatter(X_train,y_train)
plt.plot(X_train,regressor.predict(X_train))
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#plot test result
plt.scatter(X_test,y_test)
plt.plot(X_train,regressor.predict(X_train))
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

from sklearn import metrics
print("Mean Square Error :- ",metrics.mean_squared_error(y_test,pred))
print("RMSE :- ",np.sqrt(metrics.mean_squared_error(y_test,pred)))
print("R2 Score :- ",metrics.r2_score(y_test,pred))