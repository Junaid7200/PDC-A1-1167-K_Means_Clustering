#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_POINTS 1000000   // A much larger dataset (adjust as needed)
#define DIM 2               // 2D points (x and y)
#define K 3                 // Number of clusters
#define MAX_ITER 100        // Maximum iterations

// Function to compute squared Euclidean distance between two points.
double distance_sq(double p1[], double p2[]) {
    double sum = 0.0;
    for (int d = 0; d < DIM; d++) {
        double diff = p1[d] - p2[d];    // the X2 - X1 step
        sum += diff * diff;             // the sqauring
        // we don't need to take sqrt as sqrt is a monotonic operation and all we are doing here is comparing so the result of the comparison won't chang
    }
    return sum;
}

int main() {

    int i, j, iter; // we can initialize these inside the loop too as usual btw

// ===================================================================================================================================
// This section is focusing on loading/creating the data that we need to run K-Means clustering algorithm on. I wanted the amount of data to be dynamic so this is why this section is quite complicated since dynamic arrays are quite complicated in C.

    
    // Allocate memory for our dataset: an array of pointers to each point.
    double **data = malloc(NUM_POINTS * sizeof(double *));  // I don't want to give a size to the array, i want it dynamic so thats why its a pointer array.
    // its a pointer to a pointer of a double. and then we multiply the number of enteries to the size of a double and then malloc is used to allocate that much space.

    for (i = 0; i < NUM_POINTS; i++) {  // now inside each index of the data pointer array, we allocate space for each individual entry. we go through each pointer in our data array and allocate an array of DIM doubles
        data[i] = malloc(DIM * sizeof(double));
    }
    
    // Generate random data points in the range [0, 1] for each dimension.
    for (i = 0; i < NUM_POINTS; i++) {
        for (j = 0; j < DIM; j++) {
            data[i][j] = (double)rand() / RAND_MAX;
        }
    }
    
    // Array for cluster assignments (labels) for each point.
    int *labels = calloc(NUM_POINTS, sizeof(int));  // Initializes to 0.
    
    // Initialize centroids (K x DIM). We'll use the first K points as our initial centroids.
    double centroids[K][DIM];
    for (i = 0; i < K; i++) {
        for (j = 0; j < DIM; j++) {
            centroids[i][j] = data[i][j];
        }
    }
// ===================================================================================================================================

    
// ===================================================================================================================================
// This is the main section of the code where we implement K-Means clustering and this is also where we will later go about implementing parallel processing. For now, lets just do it all sequentially


    // Timing: going to use omp to calculate time of execution
    double start_time = omp_get_wtime();
    
// The logic of K-Means here is identical to the "Unnderstanding_KMeans.c" file and that file has more detailed comments too since I was understanding it there.

    int changed = 1;  // Flag to check if any point changes its cluster.
    for (iter = 0; iter < MAX_ITER && changed; iter++) {
        changed = 0;
        // Assignment Step: assign each point to the nearest centroid.
        for (i = 0; i < NUM_POINTS; i++) {
            int best_cluster = 0;
            double best_dist = distance_sq(data[i], centroids[0]);
            for (j = 1; j < K; j++) {
                double d = distance_sq(data[i], centroids[j]);
                if (d < best_dist) {
                    best_dist = d;
                    best_cluster = j;
                }
            }
            if (labels[i] != best_cluster) {
                labels[i] = best_cluster;
                changed = 1;
            }
        }
        
        // Update Step: recompute centroids as the mean of points in each cluster.
        double new_centroids[K][DIM] = {0};  // Temporary sums for each centroid.
        int counts[K] = {0};                 // Number of points in each cluster.
        
        // Sum the coordinates for each cluster.
        for (i = 0; i < NUM_POINTS; i++) {
            int cluster = labels[i];
            counts[cluster]++;
            for (j = 0; j < DIM; j++) {
                new_centroids[cluster][j] += data[i][j];
            }
        }
        // Calculate the mean (average) for each centroid.
        for (i = 0; i < K; i++) {
            if (counts[i] > 0) {  // Avoid division by zero.
                for (j = 0; j < DIM; j++) {
                    centroids[i][j] = new_centroids[i][j] / counts[i];
                }
            }
        }
    }
    
    // Timing: end the clock.
    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
// ===================================================================================================================================


// ===================================================================================================================================

// This section is just for printing out the output of the program, we won't do any parallelization here etc.

    // Print out the results.
    printf("K-Means converged in %d iterations.\n", iter);
    printf("Elapsed time (sequential): %f seconds\n", elapsed);
    printf("Final centroids:\n");
    for (i = 0; i < K; i++) {
        printf("Cluster %d: ", i);
        for (j = 0; j < DIM; j++) {
            printf("%f ", centroids[i][j]);
        }
        printf("\n");
    }
// ===================================================================================================================================


// ===================================================================================================================================

// This is another section of the code that we won't care about much during the parallelization process since its just for deallocating the memory we were using. It has nothing to do with PDC more or less.

    // Free allocated memory.
    for (i = 0; i < NUM_POINTS; i++) {
        free(data[i]);
    }
    free(data);
    free(labels);
    
    return 0;
// ===================================================================================================================================


}
