import mykmeanssp
import argparse
import math
import sys
import numpy as np
import pandas as pd


MAX_ITER = 300


def invalid_input():
    print("Invalid Input!")
    sys.exit()


def general_error():
    print("An Error Has Occurred")
    sys.exit()


def isfloat(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("k")
    parser.add_argument("maxIteration", nargs='?', default=300, const=0)
    parser.add_argument("epsilon")
    parser.add_argument("file_name_1", type=str)
    parser.add_argument("file_name_2", type=str)
    args = parser.parse_args()
    k = args.k
    if not (k.isdigit()):
        invalid_input()
    k = int(k)
    max_iter = args.maxIteration
    if max_iter != 300:
        if not (max_iter.isdigit()):
            invalid_input()
        max_iter = int(max_iter)

    if max_iter <= 0:
        invalid_input()
    epsilon = args.epsilon
    if not (isfloat(epsilon)):
        invalid_input()
    epsilon = float(epsilon)
    if epsilon < 0:
        invalid_input()
    file_name_1 = args.file_name_1
    file_name_2 = args.file_name_2
    return KMeans(k, file_name_1, file_name_2, epsilon, max_iter)


class KMeans:
    """
    a simple implementation of the K-Means++ algorithm.
    @authors: Mohammad Daghash & Ram Elgov.
    """

    def __init__(self, K, file_name_1, file_name_2, epsilon, max_iter=MAX_ITER):
        self.k = K  # number of clusters
        self.max_iter = max_iter  # maximum number of iteration for the algorithm
        self.file_name_1 = file_name_1  # an input file with valid format of data points (text file)
        self.file_name_2 = file_name_2  # an input file to save the results into (text file)
        self.epsilon = epsilon  # the accepted error
        self.data_points = pd.array([])
        self.initialize_data_points()
        self.number_of_rows = self.data_points.shape[0]
        self.number_of_cols = self.data_points.shape[1]
        self.data_points = self.data_points.to_numpy()
        if not (1 < self.k < self.number_of_rows):
            invalid_input()
        self.centroids = pd.array([])
        self.centroids_indices = []
        self.output = None
        self.D = np.array([])
        self.P = np.array([])

    def initialize_data_points(self, ):
        """
        merge the two input files
        :return:
        """
        input_1 = pd.read_csv(self.file_name_1, header=None)
        input_2 = pd.read_csv(self.file_name_2, header=None)
        self.data_points = pd.merge(input_1, input_2, how="inner", left_on=input_1.columns[0],
                                    right_on=input_2.columns[0])
        self.data_points.sort_values(by=self.data_points.columns[0], inplace=True)
        self.data_points.drop(self.data_points.columns[0], axis=1, inplace=True)
        # look https://moodle.tau.ac.il/mod/forum/discuss.php?d=104697 in the forum

    def find_min_distance(self, data_point):
        """
        a function to compute the minimal distance between a given data frame to the current existing
        centroids. assumes centroids is not empty.
        param data_point: a given data point to compute the minimal distance for.
        :return: the minimal distance between the input and the centroids.
        """
        m = math.inf
        for i in range(self.centroids.shape[0]):
            m = min(m, np.sum(np.power(np.subtract(data_point, self.centroids[i]), 2)))
        return m

    def k_means_pp(self):
        """
        an implementation of the kmeans++ algorithm to generate initial centroids
        for the use of a kmeans clustering algorithm implementation.
        assumes initialize_centroids() was already called.
        :return: a float type data frame that contains the randomly chosen centroids.
        """
        np.random.seed(0)
        miu1_index = np.random.choice(range(self.number_of_rows))
        self.centroids = np.array([self.data_points[miu1_index]])
        self.centroids_indices.append(miu1_index)
        for i in range(1, self.k):
            self.D = np.array([self.find_min_distance(self.data_points[curr])
                               for curr in range(self.number_of_rows)])
            sum_d = np.sum(self.D)
            self.P = np.array([d / sum_d for d in self.D])
            random_centroid_i = np.random.choice(range(self.number_of_rows), p=self.P)
            self.centroids = np.append(self.centroids,
                                       np.array([self.data_points[random_centroid_i]]), axis=0)
            self.centroids_indices.append(random_centroid_i)


class Error(Exception):
    pass


def print_centroid_indices(km):
    print(','.join([f"{int(i)}" for i in km.centroids_indices]))


def print_output_centroids(km):
    print(",".join(["{:.4f}".format(centroid) % centroid for centroid in km.output]))


def main():
    km = parse_input()
    km.k_means_pp()
    print_centroid_indices(km)
    print(type(km.centroids))
    print(km.centroids)
    FinalCentroids = mykmeanssp.fit(km.k, km.max_iter, km. number_of_rows, km.number_of_cols, km.epsilon,
                   km.data_points.tolist(), km.centroids.tolist())
    assert FinalCentroids, "An Error Has Occurred"
    for i in range(km.k):  # printing the final centroids
      for j in range(km.number_of_cols):
        if j != km.number_of_cols - 1:
            print(str("%.4f" % FinalCentroids[i][j]) + ",", end='')
        else:
            print(str("%.4f" % FinalCentroids[i][j]))

if __name__ == '__main__':
    try:
        main()
    except Error:
        general_error()
