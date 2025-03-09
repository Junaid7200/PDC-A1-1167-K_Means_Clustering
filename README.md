# PDC-A1-1167-K_Means_Clustering
This is a repo for the first assignment of parallel and distributed computing. The assignment revolves around choosing any kind of loop-intensive code and then implementing parallel processing to it and then doing an analysis of the potential improvements. The problem/algorithm I chose to parallelize is called K-means clustering.

## Overview
K-Means clustering is an unsupervised learning algorithm used to partition a set of data points into K clusters. The basic steps are as follows:

- Select K: Choose the number of clusters.
- Initialize Centroids: Randomly select K data points as the initial centroids.
- Assign Points: For each data point, compute the distance to each centroid (using the Euclidean distance) and assign the point to the nearest centroid.
- Update Centroids: Calculate the mean (average) of all points assigned to each cluster, and update the centroids.
- Repeat: Continue the assignment and update steps until the cluster assignments no longer change or a maximum number of iterations is reached.