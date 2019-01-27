#https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba
'''In statistics, an outlier is an observation point that is distant from other observations.'''

import pandas as pd
from sklearn.datasets import load_boston
import seaborn as sns
import matplotlib.pyplot as plt 


boston=load_boston()
x=boston.data
y=boston.target
columns=boston.feature_names

df=pd.DataFrame(boston.data)
df.columns=columns

'''In descriptive statistics, a box plot is a method for graphically depicting groups
 of numerical data through their quartiles. Box plots may also have lines extending 
vertically from the boxes (whiskers) indicating variability outside the upper and 
lower quartiles, hence the terms box-and-whisker plot and box-and-whisker diagram.
Outliers may be plotted as individual points.'''

sns.boxplot(x=df['DIS'])

'''A scatter plot , is a type of plot or mathematical diagram using Cartesian 
coordinates to display values for typically two variables for a set of data.
The data are displayed as a collection of points, each having the value of one 
variable determining the position on the horizontal axis and the value of the other 
 variable determining the position on the vertical axis.'''

plt.scatter(df['INDUS'],df['TAX'])
plt.show()

#Discover outliers with mathematical function
'''The Z-score is the signed number of standard deviations by which the value of an observation or
data point is above the mean value of what is
being observed or measured.'''

import numpy as np
from scipy import stats
z = np.abs(stats.zscore(df))
print(z)
print(np.where(z>3))
print(z[55][1])#outlier

#remove outlier and create new Data frame
df_out=df[(z<3).all(axis=1)]