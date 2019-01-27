
from sklearn.datasets import load_iris
import pandas as pd
iris=load_iris()


column=iris.feature_names
X=iris.data
Y=iris.target



df=pd.DataFrame(iris.data)
df.columns=column

#Standard Scaler
'''Standardization refers to shifting the distribution of each attribute to have a mean of 
and a standard deviation of one (unit variance).
It is useful to standardize attributes for a model that relies on the distribution of attributes 
such as Gaussian processes.'''



from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_sc=scaler.fit_transform(X)

#Data Normalization
'''Normalization refers to rescaling real valued numeric attributes into the range 0 and 1.
It is useful to scale the input attributes for a model that relies on
the magnitude of values, such as distance measures used in k-nearest neighbors
and in the preparation of coefficients in regression.'''

from sklearn.preprocessing import Normalizer
normlizer=Normalizer()
X_nz=normlizer.fit_transform(X)