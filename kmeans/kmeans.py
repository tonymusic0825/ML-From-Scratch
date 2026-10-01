"""
Last Edited: 2026-10-01
Author: Youngsu Choi

K-Means clustering implementation

"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import random

def show_kmeans(data, centers=[]):

    if centers is None or len(centers) == 0:
        centers = np.array([])

    plt.scatter(data[:, 0], data[:, 1])

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

def pairwise_euclidean_dist(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute the pairwise Euclidean distance between two given points X and Y

    Parameters
    ----------


    Raises
    ------
    ValueError
        If the shapes of two given points do not match

    """

    if X.shape != Y.shape:
        raise ValueError(
            f"Dimension mismatch: X has {len(X)} features, "
            f"while Y has {len(Y)} features"
        )
    
    D = X - Y
    
    return np.sqrt(np.dot(D, D))


class Kmeans():

    CONVERGENCE_THRESHOLD = 1e-4
    MAX_ITER = 1000

    def __init__(self, k):
        self.k = k
        self.max_iter = self.MAX_ITER
        self.data = None
        self.centers = None
    
    def load_data(self, path):
        """Loads data from CSV file

        Example
        -------
        Each row in the csv is expected to represent a single data point.

        E.g. 1, 1, 2, 3
             2, 4, 2, 1
             3, 3, 3, 3

        Represents 3 data points in 4 dimensions resulting in data shape of [X, 4]

        WARNING
        ------
        This function does not check data shape integrity. 
        Error will only flagged during euclidean distance calculation

        Raises
        ------
        ValueError
            If any non-numeric other than comma is detected
        
        """

        data = []

        with open(path) as f:
            for point in f:
                data.append(point.split(','))

                # Cast to float
                for i in range(len(data[-1])):
                    try:
                        data[-1][i] = float(data[-1][i])
                    except ValueError:
                        print(f"Data at index {len(data)}. {i} = {data[-1][i]} which is not a number.")
                

            self.data = np.array(data)

            f.close()
        
        self.create_centers()
    
    def create_centers(self):
        """Creates centers with k and data.

        WARNING
        ------- 
        Data must be filled!

        """

        if self.data is None:
            raise ValueError(
                "No data has been loaded"
            )
        
        centers = []
        rng = np.random.default_rng()

        for i in range(self.k):
            centers.append(rng.choice(self.data))
        
        self.centers = np.array(centers)

    def visualize_data(self, v_centers):
        """Visualises data using scatter plot
        
        WARNING
        -------
        This function only works for 2-D data
        
        """

        if self.data is None or self.data.shape[1] != 2:
            return
        
        if v_centers:
            show_kmeans(self.data, self.centers)
        else:
            show_kmeans(self.data) 
    
    def generate_data(
        self, n_clusters=4, n_samples_per_cluster=64,
        n_features=2, spread_factor=0.35,
        seed=None
    ):
        """Generates random cluster data"""

        if seed is None:
            seed = random.randint(1, 1000)

        rng = np.random.default_rng(seed)
        centers = rng.uniform(-5.0, 5.0, size=(n_clusters, n_features))

        diffs = centers[:, None, :] - centers[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        # Mask out self-distance (diagonal zeros) to get distance to other centers
        np.fill_diagonal(dists, np.nan)
        avg_min_dist = np.nanmean(np.nanmin(dists, axis=1))

        cluster_std = avg_min_dist * spread_factor
        noise = rng.standard_normal((n_clusters, n_samples_per_cluster, n_features))
        points = centers[:, None, :] + noise * cluster_std

        X = points.reshape(-1, n_features)
        
        self.data = X
        self.create_centers()
    
    def has_converged(self, new_centers):

        return False
        
    def fit(self):
        """Performs K-means. If no data was loaded random data will be generated

        """

        if self.data is None:
            self.data = generate_data() 
        
        count = 0

        while True:
            converged = True
            
            # Euclidean dist
            diff = self.centers[:, None, :] - self.data[None, :, :]
            diff = np.sum(np.square(diff), axis=2)
            diff = np.argmin(diff, axis=0) 

            # Update centers
            for i in range(self.k):
                
                bool_a = diff == i
                bool_data = self.data[bool_a]

                if len(bool_data) == 0:
                    continue

                new_center = np.mean(bool_data, axis=0)

                # Check for convergence
                diff_bool = np.abs((self.centers[i] - new_center)) > self.CONVERGENCE_THRESHOLD

                if np.sum(diff_bool) > 0:
                    converged = False

                self.centers[i] = new_center

            count += 1

            if converged or count > self.MAX_ITER:
                print(f"Iterated {count} times")
                break



