#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


// ===================================================================================================================================

// same as before

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

// ===================================================================================================================================



int main(int argc, char *argv[]) {

    int num_threads = atoi(argv[1]);
    omp_set_num_threads(num_threads);

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

// This is the section of the code where we will be doing our parallelization:


    int changed = 1;  // Flag to check if any point changes its cluster.

 // Each iteration depends on the results of the previous iteration, so you cannot fully parallelize across iter. We won't try to parallelize this loop
    for (iter = 0; iter < MAX_ITER && changed; iter++) {
        changed = 0;
        // Assignment Step: assign each point to the nearest centroid.
        // in the following loop, we are accessing the data points (rows of the 2D array you can say) and hence we can use diff threads to access these data points separately, so parallel for can be used here:
        #pragma omp parallel for private(j) reduction(|:changed)
        for (i = 0; i < NUM_POINTS; i++) {  // ASSIGNMENT STEP
            int best_cluster = 0;   
            double best_dist = distance_sq(data[i], centroids[0]);  // no race condition here since its local, and no race condition on i either since the for directive takes care of that on its own.
                for (j = 1; j < K; j++) {   // there will be a race condition on j since it is initialized outside this region so its not local, this is why we made it private in the directive above
                double d = distance_sq(data[i], centroids[j]);
                if (d < best_dist) {
                    best_dist = d;
                    best_cluster = j;
                }
            }
            if (labels[i] != best_cluster) {
                labels[i] = best_cluster;
                changed |= 1;
            }
        }
        
        // Update Step: recompute centroids as the mean of points in each cluster.
        double new_centroids[K][DIM] = {0};  // Temporary sums for each centroid.
        int counts[K] = {0};                 // Number of points in each cluster.
        
// ===================================================================================================================================

// this was the logic for the summation step before, i have changed it quite a bit now since this logic had 3 potential race conditions and using atomic wasn't working here, neither was private.

        // Sum the coordinates for each cluster.
        // #pragma omp parallel for
        // for (i = 0; i < NUM_POINTS; i++) {  // SUMMATION STEP
        //     int cluster = labels[i];
        //     counts[cluster]++;                              // race condition
        //     for (j = 0; j < DIM; j++) {                     // race condition
        //         new_centroids[cluster][j] += data[i][j];    // race condition
        //     }
        // }

// ===================================================================================================================================


// ===================================================================================================================================
// the summation step we had before had 3 variables where we had potential race conditons, so I'm going to change the logic for them entirely to make this loop easier to parallelize.

// we will be doing partial sum here. now, there is the question that why not use reduction(+) if we are going to partial sum. Its bcz that only works on variables and not arrays and we are using arrays here.
// another question is that why would we make local copies for each thread manually like I did below when we can just use the private directive and that way each thread can have its own local copy. thats I didn't do that bcz private has no way of merging those local copies and we need to merge them at the end

// so that gets us to a manual partial sum technique which we will implement below

// we will create local versions of counts and new_centroids (these were causing race condition previously btw) and then each thread can work on its own version of counts and new_centroid.

#pragma omp parallel
{
    // Allocate thread-local arrays for partial sums and counts
    double local_new_centroids[K][DIM] = {0}; 
    int local_counts[K] = {0};

    // Each thread processes a portion of the data:
    #pragma omp for nowait  // the threads and their iterations don't depend on each other and they can merge and end when ever they wish so using nowait here is acceptable
    for (int i = 0; i < NUM_POINTS; i++) {
        int cluster = labels[i];
        local_counts[cluster]++;    // before, we had a race condition here. lets walk through it:
        // so 2 threads move into this outer loop and then they calculate their own cluster variable, say that value of i for one thread is 2 and for the other is 6. now when these 2 threads choose 2 and 6 and then move to counts...in the previous impelementation, they would move to a global counts array and move to the same index of that and race condition would occur.
        // but now, each thread has its own counts array so race condition is avoided
        for (int d = 0; d < DIM; d++) { // before, there was a j variable being used here which was declared outside the loop and was causing race condtion, now we are using d and declaring it inside the loop so race condition is gone.
            local_new_centroids[cluster][d] += data[i][d];  // before, there was a race condition on new_centroids since it was declared outside the outer loop and 2 threads could access it at the same time, but now, each thread has its own local_new_centroid and therefore, race condition is avoided.

            // so all 3 race conditions are gone now
        }
    }

    // now, above we removed all 3 of those race conditions and did our calculations in local variables, now we will be combining all those local calculations together and storing them in our non-local version of counts and new_centroids and since those are non-local, there would be a race condition on them both, so we will make each thread go in one by one (using critical directive) and add its local version to the global version, and each thread will add it and complete its journey basically. that way the total will be stored in the main versions of counts and new_centroids
    #pragma omp critical
    {
        for (int c = 0; c < K; c++) {
            counts[c] += local_counts[c];
            for (int d = 0; d < DIM; d++) {
                new_centroids[c][d] += local_new_centroids[c][d];
            }
        }
    }
}


// ===================================================================================================================================




// ===================================================================================================================================


    // we can parallelize this loop too but value of k and DIM is so small that the changes would be next to nothing
        // Calculate the mean (average) for each centroid.
        for (i = 0; i < K; i++) {   //  MEAN CALCULATIN STEP
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
