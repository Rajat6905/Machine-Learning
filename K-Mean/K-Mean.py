
import matplotlib.pyplot as plt
from sklearn import datasets
iris=datasets.load_iris()
data=iris.data

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(data)
data=sc.transform(data)


from sklearn.cluster import KMeans
cl=KMeans(n_clusters=3)
cl.fit(data)


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i,  max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
    
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()


plt.scatter(data[:,0],data[:,1],c=cl.labels_)
plt.scatter(cl.cluster_centers_[:,0],cl.cluster_centers_[:,1],c='R',marker='X')
plt.show()






