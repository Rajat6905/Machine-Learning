import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
iris=datasets.load_iris()
X=iris.data

colors = 10*["g","r","c","b","k"]
class K_Means():
    def __init__(self,k=3,tol=0.001,max_iter=300):
        self.k=k
        self.tol=tol
        self.max_iter=max_iter
    def fit(self,data):
        self.centroids={}
        
        for i in range(self.k):
            self.centroids[i]=data[i]
    
        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []
            for featureset in data:
                    distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                    classification = distances.index(min(distances))
                    self.classifications[classification].append(featureset)
            prev_centroids = dict(self.centroids)
        
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
        
#        print(self.classifications)
            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
#                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False
            if optimized:
                break
#        print(self.classifications)
    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
               
    
clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:

    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],c='black',marker='X')

for classification in clf.classifications:
    
    
    for featureset in clf.classifications[classification]:
        print(featureset[0])
        plt.scatter(featureset[0], featureset[1],c=colors[classification])
        plt.xlabel('sepal length')
        plt.ylabel('sepal width')
    
        
        
plt.show()