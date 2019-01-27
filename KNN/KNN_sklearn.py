from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train )
iris_pred=classifier.predict(X_test)  



from sklearn.metrics import confusion_matrix,accuracy_score
#find optimal value of k
s=[]
for k in range(1,26):
    classifier=KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train,y_train )
    iris_pred=classifier.predict(X_test) 
    s.append(accuracy_score(y_test,iris_pred))


import matplotlib.pyplot as plt
plt.plot(range(1,26),s)



print(confusion_matrix(y_test,iris_pred))
print(accuracy_score(y_test,iris_pred))