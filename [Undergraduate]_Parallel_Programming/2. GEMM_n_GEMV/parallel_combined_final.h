#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <numeric>
// You cannot use OpenMP <omp.h>
// Include header files if you need,
// but it must work without modifying the Makefile
//////////////////// add: header ////////////////////
#include <thread>
#include <vector>
#include <iostream>
#include <cstring>
#include <random>
/////////////////////////////////////////////////////

/**
 * @brief Initializes a vector with random double values.
 *
 * This function fills the given array
 *
 * @param a Pointer to the array to be initialized.
 * @param N The number of elements in the array.
 */
inline void init_vec(double *a, int N) {
        /****************/
        /* TODO: put your own parallelized code here */
        /* You don't have to parallelize all of your code - it's up to you. */
        /////////////////////////////////////////////////////////////////////
//        const int num_threads = 32;
//        std::thread threads[num_threads];
//
//        auto worker = [&](int start, int end, unsigned int seed) {
//          std::mt19937 gen(seed);
//            std::uniform_int_distribution<int> dist(0, 1);
//            for (int i = start; i < end; ++i) {
//                // a[i] = 1.0 + (i % 7) * 0.123;
//                // a[i] = 0;
//                // a[i] = 1.0;
//                // a[i] = 1.0 + (i % 10) * 0.1;
//                // a[i] = ( (i & 1) ? 0.0 : 1.0 );
//              a[i] = static_cast<double>(dist(gen));
//            }
//        };
//
//        int chunk_size = N / num_threads;
//
//        for (int t = 0; t < num_threads; ++t) {
//            int start = t * chunk_size;
//            int end = std::min(start + chunk_size, N);
//            threads[t] = std::thread(worker, start, end, static_cast<unsigned int>(std::random_device{}()));
//        }
//
//        for (auto &th : threads) th.join();
        /////////////////////////////////////////////////////////////////////
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(0, 1);
        for (int i = 0; i < N; ++i) {
            // a[i] = 1.0 + (i % 7) * 0.123;
            // a[i] = 0;
            // a[i] = 1.0;
            // a[i] = 1.0 + (i % 10) * 0.1;
            // a[i] = ( (i & 1) ? 0.0 : 1.0 );
            a[i] = static_cast<double>(dist(gen));
        }
        /////////////////////////////////////////////////////////////////////
        /****************/
}

/**
 * @brief Performs a matrix-vector multiplication.
 *
 * This function computes the product of a matrix 'a' and a vector 'b', storing
 * the result in vector 'c'.
 *
 * @param a Pointer to the first element of the matrix 'a' (assumed to be in
 * row-major order).
 * @param b Pointer to the first element of the vector 'b'.
 * @param c Pointer to the first element of the result vector 'c'.
 * @param N The dimension of the matrix and vectors (assuming a square matrix
 * and compatible vector sizes).
 */
inline void gemv(double *a, double *b, double *c, int N) {
        /****************/
        /* TODO: put your own parallelized code here */
        /* You don't have to parallelize all of your code - it's up to you. */
        /////////////////////////////////////////////////////////////////////
        const int num_threads = 16;
        std::thread threads[num_threads];

        const int B = 64;  // block size

        ///////////////////// Not using Block ////////////////////////////
        // auto worker = [&](int i_start, int i_end) {
        //     for (int i = i_start; i < i_end; ++i) {
        //         double sum = 0.0;
        //         int j = 0;

        //         // Loop Unrolling
        //         for (; j + 7 < N; j += 8) {
        //             sum += a[i * N + j] * b[j];
        //             sum += a[i * N + j + 1] * b[j + 1];
        //             sum += a[i * N + j + 2] * b[j + 2];
        //             sum += a[i * N + j + 3] * b[j + 3];
        //             sum += a[i * N + j + 4] * b[j + 4];
        //             sum += a[i * N + j + 5] * b[j + 5];
        //             sum += a[i * N + j + 6] * b[j + 6];
        //             sum += a[i * N + j + 7] * b[j + 7];
        //         }
        //         for (; j < N; ++j) {
        //             sum += a[i * N + j] * b[j];
        //         }

        //         c[i] = sum;
        //     }
        // };
        ///////////////////////// Using Block (loop: ii -> i -> jj -> j) ////////////////////////////
        // auto worker = [&](int i_start, int i_end) {
        //     for (int ii = i_start; ii < i_end; ii += B) {
        //     int i_max = std::min(ii + B, i_end);

        //     for (int i = ii; i < i_max; ++i) {
        //         double sum = 0.0;

        //         for (int jj = 0; jj < N; jj += B) {
        //             int j_max = std::min(jj + B, N);
        //             int j = jj;

        //             // Loop unrolling (by 32)
        //             for (; j + 31 < j_max; j += 32) {
        //                 sum += a[i * N + j + 0] * b[j + 0];
        //                 sum += a[i * N + j + 1] * b[j + 1];
        //                 sum += a[i * N + j + 2] * b[j + 2];
        //                 sum += a[i * N + j + 3] * b[j + 3];
        //                 sum += a[i * N + j + 4] * b[j + 4];
        //                 sum += a[i * N + j + 5] * b[j + 5];
        //                 sum += a[i * N + j + 6] * b[j + 6];
        //                 sum += a[i * N + j + 7] * b[j + 7];
        //                 sum += a[i * N + j + 8] * b[j + 8];
        //                 sum += a[i * N + j + 9] * b[j + 9];
        //                 sum += a[i * N + j + 10] * b[j + 10];
        //                 sum += a[i * N + j + 11] * b[j + 11];
        //                 sum += a[i * N + j + 12] * b[j + 12];
        //                 sum += a[i * N + j + 13] * b[j + 13];
        //                 sum += a[i * N + j + 14] * b[j + 14];
        //                 sum += a[i * N + j + 15] * b[j + 15];
        //                 sum += a[i * N + j + 16] * b[j + 16];
        //                 sum += a[i * N + j + 17] * b[j + 17];
        //                 sum += a[i * N + j + 18] * b[j + 18];
        //                 sum += a[i * N + j + 19] * b[j + 19];
        //                 sum += a[i * N + j + 20] * b[j + 20];
        //                 sum += a[i * N + j + 21] * b[j + 21];
        //                 sum += a[i * N + j + 22] * b[j + 22];
        //                 sum += a[i * N + j + 23] * b[j + 23];
        //                 sum += a[i * N + j + 24] * b[j + 24];
        //                 sum += a[i * N + j + 25] * b[j + 25];
        //                 sum += a[i * N + j + 26] * b[j + 26];
        //                 sum += a[i * N + j + 27] * b[j + 27];
        //                 sum += a[i * N + j + 28] * b[j + 28];
        //                 sum += a[i * N + j + 29] * b[j + 29];
        //                 sum += a[i * N + j + 30] * b[j + 30];
        //                 sum += a[i * N + j + 31] * b[j + 31];
        //             }

        //             // Remaining j
        //             for (; j < j_max; ++j) {
        //                 sum += a[i * N + j] * b[j];
        //             }
        //             c[i] = sum;
        //         }
        //     }
        // }
        // };
        //////////////// Not using block for j & use Unrolling (loop: ii -> i -> j) ///////////////////
        // auto worker = [&](int i_start, int i_end) {
        //     for (int ii = i_start; ii < i_end; ii += B) {
        //     int i_max = std::min(ii + B, i_end);

        //     for (int i = ii; i < i_max; ++i) {
        //             double sum = 0.0;
        //             int j = 0;

        //             // Loop unrolling (by 8)
        //             for (; j + 7 < N; j += 8) {
        //                 sum += a[i * N + j + 0] * b[j + 0];
        //                 sum += a[i * N + j + 1] * b[j + 1];
        //                 sum += a[i * N + j + 2] * b[j + 2];
        //                 sum += a[i * N + j + 3] * b[j + 3];
        //                 sum += a[i * N + j + 4] * b[j + 4];
        //                 sum += a[i * N + j + 5] * b[j + 5];
        //                 sum += a[i * N + j + 6] * b[j + 6];
        //                 sum += a[i * N + j + 7] * b[j + 7];
        //             }

        //             // calculate remaining j
        //             for (; j < N; ++j) {
        //                 sum += a[i * N + j] * b[j];
        //             }
        //             c[i] = sum;
        //         }
        //     }
        // };
        ////////////////////////// Using Block (loop: ii -> jj -> i -> j) /////////////////////////
        // const int Ti = 32;  // block size
        // const int Tj = 128;  // block size

        // auto worker = [&](int i_start, int i_end) {
        //     /***** Block Matrix multiplication *****/
        //     for (int ii = i_start; ii < i_end; ii += Ti) {
        //         int i_max = std::min(ii + Ti, i_end);

        //         for (int jj = 0; jj < N; jj += Tj) {
        //             int j_max = std::min(jj + Tj, N);

        //             for (int i = ii; i < i_max; ++i) {
        //                 double sum = 0.0;

        //                 // Loop unrolling (by 32)
        //                 int j = jj;
        //                 for (; j + 31 < j_max; j += 32) {
        //                     sum += a[i * N + j + 0] * b[j + 0];
        //                     sum += a[i * N + j + 1] * b[j + 1];
        //                     sum += a[i * N + j + 2] * b[j + 2];
        //                     sum += a[i * N + j + 3] * b[j + 3];
        //                     sum += a[i * N + j + 4] * b[j + 4];
        //                     sum += a[i * N + j + 5] * b[j + 5];
        //                     sum += a[i * N + j + 6] * b[j + 6];
        //                     sum += a[i * N + j + 7] * b[j + 7];
        //                     sum += a[i * N + j + 8] * b[j + 8];
        //                     sum += a[i * N + j + 9] * b[j + 9];
        //                     sum += a[i * N + j + 10] * b[j + 10];
        //                     sum += a[i * N + j + 11] * b[j + 11];
        //                     sum += a[i * N + j + 12] * b[j + 12];
        //                     sum += a[i * N + j + 13] * b[j + 13];
        //                     sum += a[i * N + j + 14] * b[j + 14];
        //                     sum += a[i * N + j + 15] * b[j + 15];
        //                     sum += a[i * N + j + 16] * b[j + 16];
        //                     sum += a[i * N + j + 17] * b[j + 17];
        //                     sum += a[i * N + j + 18] * b[j + 18];
        //                     sum += a[i * N + j + 19] * b[j + 19];
        //                     sum += a[i * N + j + 20] * b[j + 20];
        //                     sum += a[i * N + j + 21] * b[j + 21];
        //                     sum += a[i * N + j + 22] * b[j + 22];
        //                     sum += a[i * N + j + 23] * b[j + 23];
        //                     sum += a[i * N + j + 24] * b[j + 24];
        //                     sum += a[i * N + j + 25] * b[j + 25];
        //                     sum += a[i * N + j + 26] * b[j + 26];
        //                     sum += a[i * N + j + 27] * b[j + 27];
        //                     sum += a[i * N + j + 28] * b[j + 28];
        //                     sum += a[i * N + j + 29] * b[j + 29];
        //                     sum += a[i * N + j + 30] * b[j + 30];
        //                     sum += a[i * N + j + 31] * b[j + 31];
        //                 }

        //                 // Remaining j
        //                 for (; j < j_max; ++j) {
        //                     sum += a[i * N + j] * b[j];
        //                 }

        //                 c[i] += sum;
        //             }
        //         }
        //     }

        // };


        // /***** Executing threads using function above *****/
        // int chunk_size = N / num_threads;

        // for (int t = 0; t < num_threads; ++t) {
        //     int start = t * chunk_size;
        //     int end = start + chunk_size;
        //     threads[t] = std::thread(worker, start, end);
        // }

        // for (auto &th : threads) th.join();

        //////////////// Using Block & Loop Unrolling by 4 (loop: ii -> jj -> i -> j) //////////////////
//        const int Ti = 64;  // block size
//        const int Tj = 64;  // block size

//        auto worker = [&](int i_start, int i_end) {
//            for (int ii = i_start; ii < i_end; ii += B) {
//                int i_max = std::min(ii + B, i_end);
//
//                for (int jj = 0; jj < N; jj += B) {
//                    int j_max = std::min(jj + B, N);
//
//                    for (int i = ii; i < i_max; ++i) {
//                        double sum = 0.0;
//
//                        // Loop unrolling (by 4)
//                        int j = jj;
//                        for (; j + 3 < j_max; j += 4) {
//                            sum += a[i * N + j + 0] * b[j + 0];
//                            sum += a[i * N + j + 1] * b[j + 1];
//                            sum += a[i * N + j + 2] * b[j + 2];
//                            sum += a[i * N + j + 3] * b[j + 3];
//                        }
//
//                        // Remaining j
//                        for (; j < j_max; ++j) {
//                            sum += a[i * N + j] * b[j];
//                        }
//
//                        c[i] += sum;
//                    }
//                }
//            }
//        };
//
//        /***** Executing threads using function above *****/
//        int chunk_size = N / num_threads;
//
//        for (int t = 0; t < num_threads; ++t) {
//            int start = t * chunk_size;
//            int end = start + chunk_size;
//            threads[t] = std::thread(worker, start, end);
//        }
//
//        for (auto &th : threads) th.join();
//////////////////// Using Block & Loop Unrolling by 4 & local B (loop: ii -> jj -> i -> j) /////////////

//      for (int i = 0; i < 10; i++)
//          std:: cout << c[i] << std::endl;
//
        std::fill(c, c + N, 0.0);

        auto worker = [&](int i_start, int i_end) {
            double b_local[B];  // make local b vector

            for (int ii = i_start; ii < i_end; ii += B) {
                int i_max = std::min(ii + B, i_end);

                for (int jj = 0; jj < N; jj += B) {
                    int j_max = std::min(jj + B, N);

                    // copy b[jj:j_max] to store in $L1 cache
                    for (int j = jj; j < j_max; ++j)
                        b_local[j - jj] = b[j];

                    for (int i = ii; i < i_max; ++i) {
                        double sum = 0.0;
                        int j = 0;

                        // Loop unrolling (by 4)
                        for (; j + 3 < j_max - jj; j += 4) {
                            sum += a[i * N + jj + j + 0] * b_local[j + 0];
                            sum += a[i * N + jj + j + 1] * b_local[j + 1];
                            sum += a[i * N + jj + j + 2] * b_local[j + 2];
                            sum += a[i * N + jj + j + 3] * b_local[j + 3];
                        }

                        for (; j < j_max - jj; ++j) {
                            sum += a[i * N + jj + j] * b_local[j];
                        }

                        c[i] += sum;
                    }
                }
            }
        };

//      for (int i = 0; i < 10; i++)
//          std:: cout << c[i] << std::endl;

        /***** Executing threads using function above *****/
        int chunk_size = N / num_threads;

        for (int t = 0; t < num_threads; ++t) {
            int start = t * chunk_size;
            int end = start + chunk_size;
            threads[t] = std::thread(worker, start, end);
        }

        for (auto &th : threads) th.join();


//      for (int i = 0; i < 10; i++)
//          std:: cout << c[i] << std::endl;
//
        ///////////////////////////////// use threads for j  /////////////////////////////////
        // /*** partial_sums vector (buffer) ***/
        // // partial_sums[t * N + i] = partial sum result: t'nd thread sum for row i
        // std::vector<double> partial_sums(num_threads * N, 0.0);

        // int chunk_size = (N + num_threads - 1) / num_threads; // ceil(N / num_threads)

        // /*** thread worker function ***/
        // auto worker = [&](int t_id, int j_start, int j_end)
        // {
        //     // need to accumulate in partial_sums[t_id*N + i]
        //     // sum locally and accumulate in partial_sums[t_id*N + i]

        //     for (int jj = j_start; jj < j_end; jj += B) {
        //         int j_max = std::min(jj + B, j_end);

        //         for (int i = 0; i < N; ++i) {
        //             double sum_local = 0.0;

        //             int j = jj;

        //             // Loop unrolling (by 16)
        //             for (; j + 15 < j_max; j += 16) {
        //                 sum_local += a[i*N + j +  0] * b[j +  0];
        //                 sum_local += a[i*N + j +  1] * b[j +  1];
        //                 sum_local += a[i*N + j +  2] * b[j +  2];
        //                 sum_local += a[i*N + j +  3] * b[j +  3];
        //                 sum_local += a[i*N + j +  4] * b[j +  4];
        //                 sum_local += a[i*N + j +  5] * b[j +  5];
        //                 sum_local += a[i*N + j +  6] * b[j +  6];
        //                 sum_local += a[i*N + j +  7] * b[j +  7];
        //                 sum_local += a[i*N + j +  8] * b[j +  8];
        //                 sum_local += a[i*N + j +  9] * b[j +  9];
        //                 sum_local += a[i*N + j + 10] * b[j + 10];
        //                 sum_local += a[i*N + j + 11] * b[j + 11];
        //                 sum_local += a[i*N + j + 12] * b[j + 12];
        //                 sum_local += a[i*N + j + 13] * b[j + 13];
        //                 sum_local += a[i*N + j + 14] * b[j + 14];
        //                 sum_local += a[i*N + j + 15] * b[j + 15];
        //             }

        //             // sum remaining j
        //             for (; j < j_max; ++j) {
        //                 sum_local += a[i*N + j] * b[j];
        //             }

        //             // accumulate in partial_sums
        //             partial_sums[t_id * N + i] += sum_local;
        //         }
        //     }
        // };

        // /***** Executing threads using function above *****/

        // for (int t = 0; t < num_threads; ++t) {
        //     int start = t * chunk_size;
        //     int end = start + chunk_size;
        //     threads[t] = std::thread(worker, t, start, end);
        // }

        // /*** thread join ***/
        // for (auto &th : threads) {
        //     th.join();
        // }

        // /*** accumulate all the partial sums and make C[i] ***/
        // // accumulate all the partial_sums[t*N + i] and store into c[i]
        // for (int i = 0; i < N; ++i) {
        //     double total = 0.0;
        //     for(int t = 0; t < num_threads; ++t) {
        //         total += partial_sums[t * N + i];
        //     }
        //     c[i] = total;
        // }
        /////////////////////////////////////////////////////////////////////
        /****************/
}

/**
 * @brief Performs matrix multiplication of two NxN matrices.
 *
 * This function computes the product of two square matrices `a` and `b`,
 * and stores the result in matrix `c`. All matrices are represented as
 * 1-dimensional arrays in row-major order.
 *
 * @param a Pointer to the first input matrix (NxN).
 * @param b Pointer to the second input matrix (NxN).
 * @param c Pointer to the output matrix (NxN) where the result will be stored.
 * @param N The dimension of the matrices (number of rows and columns).
 */
inline void gemm(double *a, double *b, double *c, int N) {
        /****************/
        /* TODO: put your own parallelized code here */
        /* You don't have to parallelize all of your code - it's up to you. */
        //////////////////// transpose b (loop: i -> j -> k) ////////////////////
        // std::vector<double> b_t(N * N);
        // for (int i = 0; i < N; ++i)
        //     for (int j = 0; j < N; ++j)
        //         b_t[j * N + i] = b[i * N + j]; // transpose B

        // const int num_threads = 32;
        // std::thread threads[num_threads];

        // auto worker = [&](int start_row, int end_row) {
        //     for (int i = start_row; i < end_row; ++i) {
        //             for (int j = 0; j < N; ++j) {
        //                 double sum = 0.0;
        //                 for (int k = 0; k < N; ++k) {
        //                     sum += a[i * N + k] * b_t[j * N + k];
        //                 }
        //                 c[i * N + j] = sum;
        //             }
        //     }
        // };

        // int chunk_size = N / num_threads;
        // for (int t = 0; t < num_threads; ++t) {
        //     int start = t * chunk_size;
        //     int end = start + chunk_size;
        //     threads[t] = std::thread(worker, start, end);
        // }

        // for (auto &th : threads) th.join();

        //////////////////// transpose b & loop unrolling (loop: i -> j -> k) ////////////////////
        // const int B = 64;  // block size (consider $L1)
        // const int num_threads = 32;
        // std::thread threads[num_threads];

        // // transpose B and store into b_t (B_t[j * N + i] == B[i * N + j])
        // std::vector<double> b_t(N * N);
        // for (int i = 0; i < N; ++i)
        //     for (int j = 0; j < N; ++j)
        //         b_t[j * N + i] = b[i * N + j];

        // auto worker = [&](int i_start, int i_end) {
        //     /***** Block Matrix multiplication *****/
        //     for (int ii = i_start; ii < i_end; ii += B) {
        //         int ii_end = std::min(ii + B, i_end);

        //         for (int jj = 0; jj < N; jj += B) {
        //             int jj_end = std::min(jj + B, N);

        //             for (int kk = 0; kk < N; kk += B) {
        //                 int kk_end = std::min(kk + B, N);

        //                 for (int i = ii; i < ii_end; ++i) {
        //                     for (int j = jj; j < jj_end; ++j) {
        //                         double sum = (kk == 0 ? 0.0 : c[i * N + j]);

        //                         int k = kk;
        //                         /***** Loop unrolling (by 32) *****/
        //                         for (; k + 31 < kk_end; k += 32) {
        //                             sum += a[i * N + k]     * b_t[j * N + k];
        //                             sum += a[i * N + k + 1] * b_t[j * N + k + 1];
        //                             sum += a[i * N + k + 2] * b_t[j * N + k + 2];
        //                             sum += a[i * N + k + 3] * b_t[j * N + k + 3];
        //                             sum += a[i * N + k + 4] * b_t[j * N + k + 4];
        //                             sum += a[i * N + k + 5] * b_t[j * N + k + 5];
        //                             sum += a[i * N + k + 6] * b_t[j * N + k + 6];
        //                             sum += a[i * N + k + 7] * b_t[j * N + k + 7];
        //                             sum += a[i * N + k + 8] * b_t[j * N + k + 8];
        //                             sum += a[i * N + k + 9] * b_t[j * N + k + 9];
        //                             sum += a[i * N + k + 10] * b_t[j * N + k + 10];
        //                             sum += a[i * N + k + 11] * b_t[j * N + k + 11];
        //                             sum += a[i * N + k + 12] * b_t[j * N + k + 12];
        //                             sum += a[i * N + k + 13] * b_t[j * N + k + 13];
        //                             sum += a[i * N + k + 14] * b_t[j * N + k + 14];
        //                             sum += a[i * N + k + 15] * b_t[j * N + k + 15];
        //                             sum += a[i * N + k + 16] * b_t[j * N + k + 16];
        //                             sum += a[i * N + k + 17] * b_t[j * N + k + 17];
        //                             sum += a[i * N + k + 18] * b_t[j * N + k + 18];
        //                             sum += a[i * N + k + 19] * b_t[j * N + k + 19];
        //                             sum += a[i * N + k + 20] * b_t[j * N + k + 20];
        //                             sum += a[i * N + k + 21] * b_t[j * N + k + 21];
        //                             sum += a[i * N + k + 22] * b_t[j * N + k + 22];
        //                             sum += a[i * N + k + 23] * b_t[j * N + k + 23];
        //                             sum += a[i * N + k + 24] * b_t[j * N + k + 24];
        //                             sum += a[i * N + k + 25] * b_t[j * N + k + 25];
        //                             sum += a[i * N + k + 26] * b_t[j * N + k + 26];
        //                             sum += a[i * N + k + 27] * b_t[j * N + k + 27];
        //                             sum += a[i * N + k + 28] * b_t[j * N + k + 28];
        //                             sum += a[i * N + k + 29] * b_t[j * N + k + 29];
        //                             sum += a[i * N + k + 30] * b_t[j * N + k + 30];
        //                             sum += a[i * N + k + 31] * b_t[j * N + k + 31];
        //                         }
        //                         for (; k < kk_end; ++k) {
        //                             sum += a[i * N + k] * b_t[j * N + k];
        //                         }
        //                         c[i * N + j] = sum;
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // };

        // /***** Executing threads using function above *****/
        // int chunk_size = N / num_threads;

        // for (int t = 0; t < num_threads; ++t) {
        //     int start = t * chunk_size;
        //     int end = start + chunk_size;
        //     threads[t] = std::thread(worker, start, end);
        // }

        // for (auto &th : threads) th.join();
        //////////////// transpose b & loop unrolling by 4 (loop: i -> j -> k) ///////////////////////
//        const int B = 64;  // block size (consider L1)
//        const int num_threads = 32;
//        std::thread threads[num_threads];
//
//        // transpose B and store into b_t (B_t[j * N + i] == B[i * N + j])
//        std::vector<double> b_t(N * N);
//        for (int i = 0; i < N; ++i)
//            for (int j = 0; j < N; ++j)
//                b_t[j * N + i] = b[i * N + j];
//
//        auto worker = [&](int i_start, int i_end) {
//            /***** Block Matrix multiplication *****/
//            for (int ii = i_start; ii < i_end; ii += B) {
//                int ii_end = std::min(ii + B, i_end);
//
//                for (int jj = 0; jj < N; jj += B) {
//                    int jj_end = std::min(jj + B, N);
//
//                    for (int kk = 0; kk < N; kk += B) {
//                        int kk_end = std::min(kk + B, N);
//
//                        for (int i = ii; i < ii_end; ++i) {
//                            for (int j = jj; j < jj_end; ++j) {
//                                double sum = (kk == 0 ? 0.0 : c[i * N + j]);
//
//                                int k = kk;
//                                /***** Loop unrolling (by 4) *****/
//                                for (; k + 3 < kk_end; k += 4) {
//                                    sum += a[i * N + k]     * b_t[j * N + k];
//                                    sum += a[i * N + k + 1] * b_t[j * N + k + 1];
//                                    sum += a[i * N + k + 2] * b_t[j * N + k + 2];
//                                    sum += a[i * N + k + 3] * b_t[j * N + k + 3];
//                                }
//                                // remaining k
//                                for (; k < kk_end; ++k) {
//                                    sum += a[i * N + k] * b_t[j * N + k];
//                                }
//                                c[i * N + j] = sum;
//                            }
//                        }
//                    }
//                }
//            }
//        };
//
//        /***** Executing threads using function above *****/
//        int chunk_size = N / num_threads;
//            for (int t = 0; t < num_threads; ++t) {
//                int start = t * chunk_size;
//                int end = start + chunk_size;
//                threads[t] = std::thread(worker, start, end);
//            }
//            for (auto &th : threads) th.join();
        /////////////////////////////////////////////////////////////////////
        //////////////// transpose b & loop unrolling by 4 & local B (loop: ii -> jj -> i -> j) ///////////////////////

        const int B = 64;  // Block size (fits $L1)
        const int num_threads = 32;
        std::thread threads[num_threads];

//      std::fill(c, c + N * N, 0.0);

        /*** Transpose B to access in row wise manner ***/
        std::vector<double> b_t(N * N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                b_t[j * N + i] = b[i * N + j];

        auto worker = [&](int i_start, int i_end) {
            /*** Make Local blocks for putting in $L1 cache ***/
            double a_block[B][B];
            double b_block[B][B];
            double c_block[B][B];

            for (int ii = i_start; ii < i_end; ii += B) {
                int ii_end = std::min(ii + B, i_end);

                for (int jj = 0; jj < N; jj += B) {
                    int jj_end = std::min(jj + B, N);

                    /***  Initialize c_block to zero ***/
                    for (int i = 0; i < B; ++i)
                        for (int j = 0; j < B; ++j)
                            c_block[i][j] = 0.0;

                    for (int kk = 0; kk < N; kk += B) {
                        int kk_end = std::min(kk + B, N);

                        /*** Load a_block ***/
                        for (int i = 0; i < B; ++i)
                            for (int k = 0; k < B; ++k)
                                a_block[i][k] = a[(ii + i) * N + (kk + k)];

                        /*** Load b_block from transposed matrix ***/
                        for (int k = 0; k < B; ++k)
                            for (int j = 0; j < B; ++j)
                                b_block[k][j] = b_t[(jj + j) * N + (kk + k)];

                        /*** Multiply and accumulate into c_block ***/
                        for (int i = 0; i < B; ++i) {
                            for (int j = 0; j < B; ++j) {
                                double sum = c_block[i][j];

                                /*** Loop unrolling (by 4) ***/
                                int k = 0;
                                for (; k + 3 < B; k += 4) {
                                    sum += a_block[i][k + 0] * b_block[k + 0][j];
                                    sum += a_block[i][k + 1] * b_block[k + 1][j];
                                    sum += a_block[i][k + 2] * b_block[k + 2][j];
                                    sum += a_block[i][k + 3] * b_block[k + 3][j];
                                }
                                for (; k < B; ++k) {
                                    sum += a_block[i][k] * b_block[k][j];
                                }

                                c_block[i][j] = sum;
                            }
                        }
                    }

                    /*** Store back to original matrix c ***/
                    for (int i = 0; i < B; ++i)
                        for (int j = 0; j < B; ++j)
                            c[(ii + i) * N + (jj + j)] = c_block[i][j];
                }
            }
        };

        /***** Executing threads using function above *****/
        int chunk_size = N / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            int start = t * chunk_size;
            int end = (t == num_threads - 1) ? N : start + chunk_size;
            threads[t] = std::thread(worker, start, end);
        }

        for (auto &th : threads) th.join();
        /****************/
}
