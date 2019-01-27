
import operator
import numpy as np
class KnnBase:
    def __init__(self, k, weights=None):
        self.k = k
        self.weights = weights
        
    def fit(self, train_feature, train_label):
        self.train_feature = train_feature
        self.train_label = train_label

    def get_neighbors(self,train_set, test_set, k):
        
        
        euc_distance = np.sqrt(np.sum((train_set - test_set)**2 , axis=1))
        # return the index of nearest neighbour
        
        return np.argsort(euc_distance)[0:k]
    
class KnnClassifier(KnnBase):

    def predict(self, test_feature_data_point):
        # get the index of all nearest neighbouring data points
        nearest_data_point_index = self.get_neighbors(self.train_feature, test_feature_data_point, self.k)
        vote_counter = {}
#        print('Nearest Data point index ', nearest_data_point_index)
        for label in set(self.train_label):
            vote_counter[label] = 0
        # add count to class that are present in the nearest neighbors data points
        for class_index in nearest_data_point_index:
            closest_lable = self.train_label[class_index]
            vote_counter[closest_lable] += 1
#        print('Nearest data point count', vote_counter)
        # return the class that has most votes
        return max(vote_counter.items(), key = operator.itemgetter(1))[0]
        
    
    def get_accuracy(y, y_pred):
        cnt = (y == y_pred).sum()
        return round(cnt/len(y), 2)



from sklearn import datasets

# load the iris data set
iris = datasets.load_iris()
#knn_iris_acc = []
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)

clf = KnnClassifier(1)
clf.fit(X_train, y_train)

iris_pred = []
for x in X_test:
    pred = clf.predict(x)
    iris_pred.append(pred)







from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(y_test,iris_pred))
print(accuracy_score(y_test,iris_pred))

    


