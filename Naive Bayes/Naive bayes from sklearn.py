from sklearn import datasets
data=datasets.load_iris()


X=data.data
y=data.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.naive_bayes import GaussianNB

clf=GaussianNB()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

from sklearn import metrics
print("Classification Report",metrics.classification_report(y_pred,y_test))
print("Confusion Metrix:\n",metrics.confusion_matrix(y_pred,y_test))
print("Accurecy Score:",metrics.accuracy_score(y_pred,y_test))