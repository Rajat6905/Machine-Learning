import pandas as pd
import numpy as np


dataset=pd.read_csv("F:\\DB\\abalone.csv")
dataset.columns=['Sex','Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
       'Viscera weight', 'Shell weight', 'Rings']
dataset['age'] = dataset['Rings']+1.5
dataset.drop(['Rings','Sex'],axis=1,inplace=True)

X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

from sklearn import metrics
print("Mean Square Error",metrics.mean_squared_error(y_test,y_pred))
print("Roor mean Square Error",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

