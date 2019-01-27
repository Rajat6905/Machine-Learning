import pandas as pd
import numpy as np
data=pd.DataFrame({'Counter':['France' ,'Spain','Germany','Spain','Germany','France','Spain','France','Germany','France'],
                   'Age':[44,27,30,38,40,35,np.nan,48,50,37],
                   'Salary':[72000,48000, 54000,61000,np.nan,58000,52000,79000,83000,67000],
                   'Purchased':['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
                   })

X = data.iloc[:, :-1].values
y = data.iloc[:, 3].values


from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

'''LabelEncoder is used to transform non-numerical labels (as long as they are hashable and comparable) 
to numerical labels.'''
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencode_X=LabelEncoder()

#Encode independent Variable
X[:,0]=labelencode_X.fit_transform(X[:,0])

'''after label encoding, we might confuse our model into thinking that a column has data
 with some kind of order or hierarchy, when we clearly don’t have it. To avoid this, we ‘OneHotEncode’
 that column.'''
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

#Encode Dependent Varibale
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)