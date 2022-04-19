# Machine learning Clustering (DBSCAN & K-MEANS)

### DBSCAN

Density-based clustering is a method of clustering high-density parts because the dots are densely clustered. To put it simply, if there are n or more points within a radius x of a certain point, it is recognized as a single cluster

![dbscan_1 50%](https://user-images.githubusercontent.com/62922310/163939578-72986049-0963-4a4c-9019-06980b900cca.png)
![dbscan_2 50%](https://user-images.githubusercontent.com/62922310/163939582-677eeec7-6df1-4284-90c5-6d52de39bd7c.png)

### K-MEANS
It is one of the representative separable clustering algorithms, and each cluster has one centroid. Each entity is assigned to the nearest centroid, and the entities assigned to the same centroid form a cluster. In order to execute the algorithm, the user must determine the number of clusters (k) in advance.

![kmeans_1 30%](https://user-images.githubusercontent.com/62922310/163939644-6b197b49-af93-41fc-bb20-4a0ab447f15f.png)
![kmeans_2 30%](https://user-images.githubusercontent.com/62922310/163939645-8ced4dea-ff9a-45e1-b138-b0b564a85696.png)
![kmeans_3 30%](https://user-images.githubusercontent.com/62922310/163939640-d17c36ac-2a36-461d-84be-51c3b95c8646.png)

## Development Environment
- python interpreter => python 3.X
- library

  scikit-learn (pip install scikit-learn)
  
  pandas (pip install pandas)
  
  numpy (pip install numpy)
  
  matplotlib (pip install matplotlib) -> view
