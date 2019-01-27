from sklearn import datasets

data=datasets.load_breast_cancer()


features=data.feature_names
target_names=data.target_names


X=data.data

y=data.target


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



from sklearn.svm import SVC
classifier=SVC(kernel='linear')
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn import metrics
print("Accurecy Score",metrics.accuracy_score(y_test,y_pred))

print("Confusion Matrix",metrics.confusion_matrix(y_test, y_pred))
