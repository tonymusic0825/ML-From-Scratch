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
CONVERGENCE_THRESHOLD = 1e-4
rng = np.random.default_rng()

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

def update_live_plot(data, centers, iteration):
    """Clears the canvas, redraws the data and current cluster centers."""
    plt.clf()  
    
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
    
    plt.title(f"K-Means Iteration: {iteration}")
    plt.pause(0.1)  

def calc_dist(point1, point2):
    """Euclidean distance for n-dimensions (Assumes that points given are in the same dimension)"""

    total = 0.0
    n = len(point1)
    for i in range(n):
        total = total + (point2[i] - point1[i])**2

    return math.sqrt(total) 

def generate_data():
    # Dummy Data (We will create data loading from csv in a moment)
    blob1 = np.array([2, 14]) + 2.0 * np.random.randn(32, 2)
    blob2 = 6 + 1.9 * np.random.randn(32, 2)
    blob3 = 12 + 2.1 * np.random.randn(32, 2)
    blob4 = 1 + 3.4 * np.random.rand(32, 2)

    return [blob1, blob2, blob3, blob4]

def kmeans(args):

    data = generate_data()

    # Parameters
    k = args.k
    k_s = {}
    CONVERGED = [False for i in range(k)]

    for i in range(k):
        rand_blob = rng.choice(data)
        x, y = rng.choice(rand_blob)
        k_s[i] = [x, y, 0, 0, 0]

    # Plotting
    if args.v:
        plt.ion()

    # Compare all points to all clusters
    count_op = 0
    while True:
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

                temp_x = k_s[i][CLUSTER_AVG_X] / k_s[i][CLUSTER_COUNT]
                temp_y = k_s[i][CLUSTER_AVG_Y] / k_s[i][CLUSTER_COUNT]

                # Check for convergence
                if abs(temp_x - k_s[i][CLUSTER_X]) < CONVERGENCE_THRESHOLD or abs(temp_y - k_s[i][CLUSTER_Y]) < CONVERGENCE_THRESHOLD:
                    CONVERGED[i] = True

                k_s[i][CLUSTER_X] = temp_x
                k_s[i][CLUSTER_Y] = temp_y
                k_s[i][CLUSTER_AVG_X] = 0
                k_s[i][CLUSTER_AVG_Y] = 0
                k_s[i][CLUSTER_COUNT] = 0

        count_op += 1

        if args.v:
            centers_list = [[k_s[i][CLUSTER_X], k_s[i][CLUSTER_Y]] for i in range(k)]
            centers = np.array(centers_list)
            update_live_plot(data, centers, count_op)

        if False not in CONVERGED:
            print(f"Iterated {count_op} times")
            break

    if args.v:
        plt.ioff()
        plt.show()

    if not args.v:
        centers_list = []

        for i in range(k):
            x, y = k_s[i][CLUSTER_X], k_s[i][CLUSTER_Y]
            centers_list.append([x, y])

        centers = np.array(centers_list)
        show_kmeans(data, centers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MicroFlowDiT")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--v", action="store_true")
    args = parser.parse_args()
    kmeans(args)