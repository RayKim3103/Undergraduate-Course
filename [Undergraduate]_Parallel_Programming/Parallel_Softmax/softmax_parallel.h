#include <algorithm>
#include <cmath>
#include <thread>
// Do NOT add any other headers.
using namespace std; // You can remove this line if you want.

/**
 * @brief Computes the softmax function in parallel.
 *
 * This function takes an input array of floats and computes the softmax
 * function in parallel using the specified number of threads. The result
 * is stored in the output array.
 *
 * @param in Pointer to the input array of floats.
 * @param out Pointer to the output array where the softmax results will be
 * stored.
 * @param elems The number of elements in the input array.
 */
inline void softmax_parallel(float *in, float *out, int elems) {

        const int NTHREADS = 64; // Number of threads to use.
                                                         // Yon can change it to any number you want.

        /****************/
        /* TODO: put your own parallelized softmax here */
        /* You don't have to parallelize all of your code - it's up to you. */
//      int block_size = elems / NTHREADS;

        // When we see main.cpp there are total 1<<28 elements, which is a power of 2
        // No Remainder left when NTHREADS = 64 or 128
        int block_size = elems >> 6; // 64 = 2^6, 128 = 2^7

        thread threads[NTHREADS];               // Array of threads

        ////////// 1. Not parallizing finding max computation //////////
//      float max_val = *std::max_element(in, in + elems);
        ////////////////////////////////////////////////////////////////

        // 1. Find the max value in parallel (parallelized max calculation)

        // 1-1. Create an array to hold local max values for each thread
        float local_max_values[NTHREADS];

        // Use threads to find max value
        for (int i = 0; i < NTHREADS; i++) {
            threads[i] = thread([&, i]() {
                // Calculate the local max for each thread, [cf] *max_element operation is serial
                int offset = i * block_size;
                local_max_values[i] = *max_element(in + offset, in + offset + block_size);

                /***** I tried tree reduction with Dynamic allocation but, It was too slow *****/
                /***** I think it is because of memory allocation process, which cause lots of time *****/
//                float* thread_max_values = new float[block_size]; // Dynamically allocated array
//                for (int step = 1; step < block_size; step *= 2) {
//                    for (int j = 0; j < block_size; j += step * 2) {
//                        if (j + step < block_size) {
//                            thread_max_values[j] = max(in[j], in[j + step]);
//                        }
//                    }
//                }
//                local_max_values[i] = thread_max_values[0]; // Final max value stored in the first element
//
//                delete[] thread_max_values; // Deallocate the array
            });
        }

        // Wait for all threads to finish
        for (int i = 0; i < NTHREADS; i++) {
            threads[i].join();
        }

        /***** 1-2. Combining all local max values in serial *****/
        float max_val = *max_element(local_max_values, local_max_values + NTHREADS);
        /**************************************************************/

//        // 1-2. Tree reduction to find the max value in serial
//        // Combine all local max values to get the final max value
//        for (int step = 1; step < NTHREADS; step *= 2) {
//            for (int i = 0; i < NTHREADS; i += step * 2) {
//                if (i + step < NTHREADS) {
//                    local_max_values[i] = max(local_max_values[i], local_max_values[i + step]);
//                }
//            }
//        }
//
//        max_val = local_max_values[0]; // Final max value stored in the first element

        ////////// 2. Not parallizing exponential computation ///////////
//        std::for_each(out, out + elems,
//              [&](float &x) { x = std::exp(x); });
//
        /////////////////////////////////////////////////////////////

        // 2. Apply subtract and exp() to the array in parallel
        // Do exp(in[j] - max_val) -> why? If we don't do it, It may cause overflow
        for (int i = 0; i < NTHREADS; i++) {
            threads[i] = thread([&, i]() {
                int offset = i * block_size;
                for (int j = offset; j < offset + block_size; j++) {
                    out[j] = exp(in[j] - max_val);
                }
            });
        }

        // Wait for all threads to finish
        for (int i = 0; i < NTHREADS; i++) {
            threads[i].join();
        }

        ////////// 3. Not parallizing sum computation ///////////
//        float sum = 0.0f;
//        for (int i = 0; i < elems; ++i) {
//            sum += out[i];
//        }
        ////////////////////////////////////////////////////////

        // 3. Compute the sum in parallel (manual summation)
        float sum = 0.0f;
        float local_sum_values[NTHREADS];  // Store the sum value of each thread

        // get partial sum from each thread, later I will use tree reduction to get final sum
        for (int i = 0; i < NTHREADS; i++) {
            threads[i] = thread([&, i]() {
                float local_sum = 0.0f;
                int offset = i * block_size;
                for (int j = offset; j < offset + block_size; j++) {
                    local_sum += out[j];  // Sum the exponentiated values
                }
                local_sum_values[i] = local_sum;
            });
        }

        // Wait for all threads to finish
        for (int i = 0; i < NTHREADS; i++) {
            threads[i].join();
        }

        /***** 3-2. Combine all the local sums in seral *****/
        for (int i = 0; i < NTHREADS; i++) {
            sum += local_sum_values[i];
        }
        /****************************************************/

//        // 3-2. Tree reduction to combine all the local sums to get the final sum
//        for (int step = 1; step < NTHREADS; step *= 2) {
//            for (int i = 0; i < NTHREADS; i += step * 2) {
//                if (i + step < NTHREADS) {
//                    local_sum_values[i] += local_sum_values[i + step];  // Merge two adjacent sums
//                }
//            }
//        }
//
//        sum = local_sum_values[0];  // Final sum stored in the first element
                                    //
        ////////// 4. Not parallizing normalization ///////////
//        std::for_each(out, out + elems,
//            [&](float &x) { x = x / sum; });
        //////////////////////////////////////////////////////

        // 4. Normalize the results in parallel
        for (int i = 0; i < NTHREADS; i++) {
            threads[i] = thread([&, i]() {
                int offset = i * block_size;
                for (int j = offset; j < offset + block_size; j++) {
                    out[j] /= sum;
                }
            });
        }

        // Wait for all threads to finish
        for (int i = 0; i < NTHREADS; i++) {
            threads[i].join();
        }
        /****************/
}                                                          