import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from KMeans import KMeansClustering


def clean_dataset(dataset):

    # remove duplicates
    reduced_dataset = dataset.drop_duplicates()

    try:
        # replace missing data with zeros
        reduced_dataset.fillna(value=0, inplace=True)
        return reduced_dataset
    except Exception as e:
        # Log the error
        print(f"Error: {e}")
        return reduced_dataset.DataFrame()  # Return an empty DataFrame if an error occurs


def run():
    try:
        dataset = pd.read_csv('crime_data.csv')
        # print(dataset)
        cleaned_dataset = clean_dataset(dataset)
        # print(cleaned_dataset)

        data = np.array(cleaned_dataset)  # ['Alabama' 13.2 236 58 21.2]
        # print(data[:, 1])
        # Select only the numeric columns
        numeric_dataset = cleaned_dataset.select_dtypes(include='number')
        numeric_data = np.array(numeric_dataset)  # [13.2 236 58 21.2]
        # print(numeric_data[0])
        while True:

            num_of_clusters = int(input('Enter Number of Clusters : '))
            # create KMeans object and fit the data
            kmeans = KMeansClustering(k=num_of_clusters, max_iterations=100)
            kmeans.fit(numeric_data)

            # predict labels and outliers
            labels, outliers = kmeans.predict(data, threshold=3)

            # print(labels)
            # print(outliers)
            kmeans.print_clusters()
            kmeans.print_outliers(data)

            # plot the results
            plt.scatter(data[:, 0], data[:, 1], c=labels)
            plt.scatter(outliers[:, 0], outliers[:, 1], c='r', marker='x')
            plt.show()
    except Exception as e:
        print(e)


if __name__ == '__main__':
    run()
