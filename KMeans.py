import numpy as np
#
# def calculate_euclidean_distance(list_1, list_2):
#     difference = list_1 - list_2  # (3, 50, 4)
#     return np.sqrt((difference ** 2).sum(axis=2))


def calculate_manhattan_distance(list_1, list_2):
    difference = list_1 - list_2  # (3, 50, 4)
    return abs(difference).sum(axis=2)


class KMeansClustering:

    def __init__(self, k, max_iterations=100):
        self.clusters = dict()
        self.outliers = None
        self.distances = None
        self.centroids = None
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, data):

        # randomly initialize centroids
        self.centroids = data[np.random.choice(data.shape[0], self.k, replace=False)]  # 1D array
        # print(self.centroids, end='\n++\n')

        for i in range(self.max_iterations):
            # convert 1D to 2D centroids in order to be subtracted from data (shape equality+)
            upper_dimensions_centroids = self.centroids[:, np.newaxis]  # 2D array

            # 2D all distance for each centroid
            distances = calculate_manhattan_distance(data, upper_dimensions_centroids)
            # print(distances.shape, distances, "\n\**********************")
            # assign points to nearest centroid
            labels = np.argmin(distances, axis=0)  # 1D list class for each transaction
            # print(labels.shape, labels)
            # calculate new centroids
            new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(self.k)])
            # print(new_centroids)
            # check for convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        # calculate distances from each point to its assigned centroid
        distances = calculate_manhattan_distance(data, self.centroids[:, np.newaxis])  # (3, 50, 4)
        self.distances = np.min(distances, axis=0)

    def predict(self, data, threshold=3):

        numeric_data = data[:, 1:]
        # convert 1D to 2D centroids in order to be subtracted from data (shape equality+)
        upper_dimensions_centroids = self.centroids[:, np.newaxis]  # 2D array

        # print(difference.shape) # (3, 50, 4)
        # calculate distances from each point to each centroid
        distances = calculate_manhattan_distance(numeric_data, upper_dimensions_centroids)
        # print(distances)
        # distances = calculate_euclidean_distance(data, upper_dimensions_centroids)
        # labels2 = np.argmin(distances2, axis=0)

        # assign points to nearest centroid
        labels = np.argmin(distances, axis=0)

        # create dictionary of clusters

        clusters = {}

        for i in range(len(labels)-1):
            if labels[i]+1 in clusters:
                clusters[labels[i]+1].append(data[:, 0][i])
            else:
                clusters[labels[i]+1] = [data[:, 0][i]]

        # print(clusters)
        if len(clusters) > 0:
            self.clusters = clusters
        # if (labels2 == labels).all():
        #     print("same answer")
        # identify outliers as points with distance greater than threshold * std
        std = np.std(self.distances)
        outliers = numeric_data[self.distances > threshold * std]
        if len(outliers) > 0:
            self.outliers = outliers
        return labels, outliers

    def print_clusters(self):
        print('#################')
        print(f"{self.k}-Clusters")
        print('#################')
        # print(sorted(self.clusters.keys()))
        for key in sorted(self.clusters.keys()):
            print('/***************/')
            # for key, value in cluster:
            print(f'Cluster #{key} => {self.clusters.get(key)}')

        # for key, values in self.clusters.items():
        #     print('/***************/')
        #     # for key, value in cluster:
        #     print(f'Cluster #{key} => {values}')

        print('/***************/')

    def print_outliers(self, data):

        print('#################')
        print(f"{len(self.outliers)}-Outliers")
        print('#################')
        for outlier in self.outliers:
            print('/***************/')
            for record in data:
                if np.all(np.isin(outlier, record)):
                    print(record)

            # for key, value in cluster:
            # print(f'Cluster #{key} => {values}')
            # print(outlier)
        print('/***************/')
