/* K-Means Clustering:
    1. Select the number of clusters (the value of K).
    2. randomly select K data points.
    3. Measure the distance between the first point and the three initial clusters.
    4. Assign the first point to the nearest cluster.
    5. Repeat this for all the points.
    6. Calculate the mean of each cluster.
    7. recluster based on the new mean.
    8. repeat until the clusters no longer change.
    
    How to find the value of k?
    You can just try different values of k (you can quantify its 'badness' by comparing the total variation).
    we want more total variation.
    if your data points are on a plain and not a number line, you can use the euclidean distance formula (in 2 dimensions, the euclidean distance is the same as the pythagorean theorem).
    euclidean distance in 2 dimensions = Sqrt(x^2 + y^2) and that x is basically X2 - X1 and same for y.
    if it was 3 dimensions or 4 etc, you would just add another squared variables into that equation (+ z^2 for instance)
    */

    #include <stdio.h>
    
    #define NUM_POINTS 10   // A much larger dataset (adjust as needed)
    #define DIM 2               // 2D points (x and y)
    #define K 3                 // Number of clusters
    #define MAX_ITER 100        // Maximum iterations
    
    // Our fixed dataset of 10 points in 2D.
    double data[NUM_POINTS][DIM] = {
        {1.0, 2.0},
        {1.5, 1.8},
        {5.0, 8.0},
        {8.0, 8.0},
        {1.0, 0.6},
        {9.0, 11.0},
        {8.0, 2.0},
        {10.0, 2.0},
        {9.0, 3.0},
        {6.0, 7.0}
    };
    
    // Function to compute the squared Euclidean distance between two points.
    // (We use squared distance to avoid computing the square root.)
    double distance_sq(double p1[], double p2[]) {
        double sum = 0;
        for (int d = 0; d < DIM; d++) {
            double diff = p1[d] - p2[d];    // the X2 - X1 step
            sum += diff * diff;             // the X^2 step
                                            // taking sqrt is not needed since sqrt is monotonic and we are only comparing here, there would be no difference in the results.
        }
        return sum;
    }
    
    int main() {
        int i, j, iter;
        int labels[NUM_POINTS] = {0};       // which cluster is which point in (the index would be the point and the value on that index is the cluster)
        // all the labels are 0 initially
        double centroids[K][DIM];           // what is the center of which cluster (the center of the class that we choose at the very start is called the centroid)
    
        // 1. Initialize centroids.
        // Here we choose the first K data points as our initial centroids.
        for (i = 0; i < K; i++) {
            for (j = 0; j < DIM; j++) {     // we are basically traversing the 2D array, one row at a time, the outer loop moves us into the row and the inner loop moves us through both columns (since its 2D so 2 columns representing x and y)
                centroids[i][j] = data[i][j];   //the first row and second row and third row (or you can say first, second, and third data points) are chosen as centroids)
            }
        }
    
        int changed = 1;  // A flag to check if any point changed its cluster in an iteration.
        // 2. Run the algorithm until no assignments change or we hit MAX_ITER.
        for (iter = 0; iter < MAX_ITER && changed; iter++) {    // it will only stop when max iterations are reached and the changed variable is 0
            changed = 0;
    
            // Assignment Step: For each data point, find the nearest centroid.
            for (i = 0; i < NUM_POINTS; i++) {
                int best_cluster = 0;
                double best_dist = distance_sq(data[i], centroids[0]);  // simple item comparison in array (choose the first and store it and compare it with the rest and store the best)
                for (j = 1; j < K; j++) {
                    double d = distance_sq(data[i], centroids[j]);
                    if (d < best_dist) {
                        best_dist = d;
                        best_cluster = j;
                    }
                }
                // If the point's cluster assignment changes, mark that something changed.
                if (labels[i] != best_cluster) {
                    labels[i] = best_cluster;
                    changed = 1;
                }
            }
    
            // Update Step: Recompute centroids as the mean of all points assigned to them.
            double new_centroids[K][DIM] = {0};  // Temporary array for new centroid sums.
            int counts[K] = {0};                 // Count how many points belong to each cluster to help calculate the mean later.
    
            // Sum the coordinates for points in each cluster.
            for (i = 0; i < NUM_POINTS; i++) {
                int cluster = labels[i];    // again, the index of the labels array represents the data point and the value is the cluster that that data point is in.
                counts[cluster]++;  // we increment count at the index of that cluster as this current data point that we are looping over belongs in that cluster (index in counts array represents the cluster and the value on the index represents the number of data points in that cluster)
                for (j = 0; j < DIM; j++) { // add the x coordinates of each data point and the y coordinates of each data point and store them in the new_centroid 2D array
                    new_centroids[cluster][j] += data[i][j];
                }
            }
            // Compute the average to update the centroid.
            for (i = 0; i < K; i++) {
                if (counts[i] > 0) {  // Avoid division by zero.
                    for (j = 0; j < DIM; j++) {
                        centroids[i][j] = new_centroids[i][j] / counts[i];
                    }
                }
            }
        }
    
        // Print the results.
        printf("K-Means converged in %d iterations.\n", iter);
        printf("Final centroids:\n");
        for (i = 0; i < K; i++) {
            printf("Cluster %d: ", i);
            for (j = 0; j < DIM; j++) {
                printf("%f ", centroids[i][j]);
            }
            printf("\n");
        }
        printf("Point assignments:\n");
        for (i = 0; i < NUM_POINTS; i++) {
            printf("Point %d (", i);
            for (j = 0; j < DIM; j++) {
                printf("%f ", data[i][j]);
            }
            printf(") -> Cluster %d\n", labels[i]);
        }
        return 0;
    }
    






    