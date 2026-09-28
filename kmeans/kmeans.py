"""
Last Edited: 2026-09-27

"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

CLUSTER_X = 0
CLUSTER_Y = 1
CLUSTER_AVG_X = 2
CLUSTER_AVG_Y = 3
CLUSTER_COUNT = 4

def show_kmeans(data, centers=[]):

    if centers is None or len(centers) == 0:
        centers = np.array([])

    for blobs in data:
        plt.scatter(blobs[:, 0], blobs[:, 1])

    if centers.size > 0:
        plt.scatter(
            centers[:, 0], 
            centers[:, 1], 
            c="black",       
            marker="X",      
            s=200,         
            label="Centers"
        )

    plt.show()

def calc_dist(point1, point2):
    """Euclidean distance"""

    dist_x = (point2[0] - point1[0])**2
    dist_y = (point2[1] - point1[1])**2

    return math.sqrt(dist_x + dist_y)

def kmeans(args):

    # Dummy Data (We will create data loading from csv in a moment)
    blob1 = np.array([2, 14]) + 2.0 * np.random.randn(32, 2)
    blob2 = 6 + 1.9 * np.random.randn(32, 2)
    blob3 = 12 + 2.1 * np.random.randn(32, 2)
    blob4 = 1 + 3.4 * np.random.rand(32, 2)
    data = [blob1, blob2, blob3, blob4]
    show_kmeans(data)

    # Parameters
    k = args.k
    k_s = {}

    for i in range(k):
        x, y = np.random.randn(2)
        k_s[i] = [x, y, 0, 0, 0]

    # Compare all points to all clusters
    
    opt_count = 0

    while opt_count < 20:
        for blob in data:
            for (x, y) in blob:
                min = -1
                min_dist = float("inf") 
                for key, value in k_s.items():
                    dist = calc_dist((value[CLUSTER_X], value[CLUSTER_Y]), (x, y))

                    if dist < min_dist:
                        min_dist = dist
                        min = key

                k_s[min][CLUSTER_AVG_X] += x 
                k_s[min][CLUSTER_AVG_Y] += y
                k_s[min][CLUSTER_COUNT] += 1

        for i in range(k):

            if k_s[i][CLUSTER_COUNT] != 0:
                k_s[i][CLUSTER_X] = k_s[i][CLUSTER_AVG_X] / k_s[i][CLUSTER_COUNT]
                k_s[i][CLUSTER_Y] = k_s[i][CLUSTER_AVG_Y] / k_s[i][CLUSTER_COUNT]
                k_s[i][CLUSTER_AVG_X] = 0
                k_s[i][CLUSTER_AVG_Y] = 0
                k_s[i][CLUSTER_COUNT] = 0

        opt_count += 1

    centers_list = []

    for i in range(k):
        x, y = k_s[i][CLUSTER_X], k_s[i][CLUSTER_Y]
        centers_list.append([x, y])

    centers = np.array(centers_list)
    show_kmeans(data, centers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MicroFlowDiT")
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args()
    kmeans(args)