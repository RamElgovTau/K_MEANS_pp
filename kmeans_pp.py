import numpy
import numpy as np
import pandas as pd
import sys
import math

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

parser = argparse.ArgumentParser()
parser.add_argument("k")
parser.add_argument("maxIteration", nargs='?', default=300, const=0)
parser.add_argument("epsilon")
parser.add_argument("file_name_1", type=str)
parser.add_argument("file_name_2", type=str)
args = parser.parse_args()
k = args.k
if not (k.isdigit()):
    print("Invalid Input!")
    sys.exit()
k= int(k)
maxIter = args.maxIteration
if(maxIter!=300):
    if not (maxIter.isdigit()):
        print("Invalid Input!")
        sys.exit()
    maxIter = int(maxIter)

if (maxIter <= 0):
    print("Invalid Input!")
    sys.exit()

epsilon = args.epsilon
if not (isfloat(epsilon)):
    print("Invalid Input!")
    sys.exit()
epsilon=float(epsilon)
if (epsilon < 0):
    print("Invalid Input!")
    sys.exit()


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
        self.data_points = np.array([])  # holding the data points in a dictionary
        self.initialize_data_points()  # read the given data points from the input file into the dictionary
        self.n = self.data_points.shape[0]
        if not (1 < self.k < self.n):
            invalid_input()

        self.centroids = np.array([])  # holding the centroids in a dictionary

        self.initialize_centroids()  # initializing the centroids dictionary
        self.D = np.array([])  # holding the  in a dictionary
        self.P = np.array([])

    def initialize_data_points(self, ):
        """
        merge the two input files
        :return:
        """

        self.data_points = pd.merge(pd.read_csv(self.file_name_1, header=None),
                                    pd.read_csv(self.file_name_2, header=None), "inner", 0)
        self.data_points.sort_values(by=[0], inplace=True)

    def calcDis(self, data_point, centroid):
        sum = 0
        for coord in range(len(centroid)):
            sum += pow(data_point[coord] - centroid[coord], 2)
        return sum

    def minDis(self, data_point):
        min = self.calcDis(data_point, self.centroids[0])
        for i in range(len(self.centroids)):
            dis = self.calcDis(data_point, self.centroids[i])
            if (dis < min):
                min = dis
        return min

    def k_means_pp(self):
        """
        deliver the calculated distribution to the np.random.choice() function
        :return:
        """
        i = 1
        np.random.seed(0)
        np.append(self.centroids, np.random.choice(self.centroids))  # miu1 randomly selected
        while i < self.k:
            for l in range(self.n):
                np.append(self.D, self.minDis(self.data_points[l]))
            for d in self.D:
                sum_d = np.sum(self.D)
                np.append(self.P, [d / sum_d for d in self.D])
            i += 1
            np.append(self.centroids, np.random.choice(self.n, p=self.P))


class InvalidInput(Exception):
    pass


def main():
    pass


#
# if __name__ == '__main__':
#     try:
#         main()
#     except InvalidInput:
#         print("Invalid Input!")
#     except:
#         print("An Error Has Occurred")

km = KMeans(10, "input_1_db_1.txt", "input_1_db_2.txt", 0.1)
print(km.data_points)
