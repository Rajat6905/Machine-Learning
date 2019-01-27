import pandas as pd
import numpy as np



data=pd.DataFrame({'Counter':['France' ,'Spain','Germany','Spain','Germany','France','Spain','France','Germany','France'],
                   'Age':[44,27,30,38,40,35,np.nan,48,50,37],
                   'Salary':[72000,48000, 54000,61000,np.nan,58000,52000,79000,83000,67000],
                   'Purchased':['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
                   })


X = data.iloc[:, :-1].values
y = data.iloc[:, 3].values

#Display no of null values in each column
data.isnull().sum()

#Drop the rows where at least one element is missing.
data.dropna()
#Drop the columns where at least one element is missing.
data.dropna(axis=1)

#Null values in salary Column
data['Salary'].isnull().sum()

# fill missing values with mean column values
data.fillna(data.mean(), inplace=False)


'''we’ll be using the strategy ‘mean’ and imputing along the columns, as 
imputing along the rows makes no sense here'''

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
