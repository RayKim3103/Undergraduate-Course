#include "parallel_combined_2.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdlib.h>

using namespace std;

bool is_same_vec(double *a, double *b, int N) {
        for (int i = 0; i < N; i++) {
                if (abs(a[i] - b[i]) > 1) {
                        return false;
                }
        }
        return true;
}

bool is_same_mat(double *a, double *b, int N) {
        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                        if (abs(a[i * N + j] - b[i * N + j]) > 1) {
                                return false;
                        }
                }
        }
        return true;
}

int main(int argc, char **argv) {
        const int N = 1 << 11; // Data size, and change this while debugging

        double *mat_a = new double[N * N];
        double *mat_b = new double[N * N];
        double *mat_c = new double[N * N];

        // fstream f_mat_a("/data/hw2/matrix_a.txt", ios::in);
        // fstream f_mat_b("/data/hw2/matrix_b.txt", ios::in);
        // fstream f_mat_c("/data/hw2/matrix_c.txt", ios::in);
        fstream f_mat_a("C:\minsung\2. 25_1\4. MGP\HW\HW2\local\matrix_a.txt", ios::in);
        fstream f_mat_b("C:\minsung\2. 25_1\4. MGP\HW\HW2\local\matrix_b.txt", ios::in);
        fstream f_mat_c("C:\minsung\2. 25_1\4. MGP\HW\HW2\local\matrix_c.txt", ios::in);

        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                        f_mat_a >> mat_a[i * N + j];
                        f_mat_b >> mat_b[i * N + j];
                        f_mat_c >> mat_c[i * N + j];
                }
        }

        // 1. Parallel GEMM Approach
        {
                std::chrono::duration<double> diff;
                auto start = std::chrono::steady_clock::now();
                {
                        double *gemm_out = new double[N * N];
                        gemm(mat_a, mat_b, gemm_out, N);
                        if (!is_same_mat(gemm_out, mat_c, N)) {
                                cerr << "Parallel GEMM Approach Failed" << endl;
                                return -1;
                        }
                }
                auto end = std::chrono::steady_clock::now();
                diff = end - start;
                std::cout << "Parallel GEMM Approach took " << diff.count() << " sec"
                                  << std::endl;
                cout << "Parallel GEMM Approach Passed" << endl;
        }

        // 2. Parallel Freivalds ~@~Y Algorithm
        {
                double *my_vec = new double[N];
                init_vec(my_vec, N);

                std::chrono::duration<double> diff;
                auto start = std::chrono::steady_clock::now();
                {
                        double *gemv_out1 = new double[N];
                        double *gemv_out2 = new double[N];
                        double *buf = new double[N];
                        ///////////////////////////////////////////
                        // for (int i = 0; i < 10; i++) {
                        //         std::cout << buf[i] << " ";
                        // }
                        // std::cout << '\n';
                        // for (int i = 0; i < 10; i++) {
                        //         std::cout << gemv_out1[i] << " ";
                        // }
                        // std::cout << '\n';
                        // for(int i = 0; i < 10; i++) {
                        //         std::cout << gemv_out2[i] << " ";
                        // }
                        // std::cout << '\n';
                        ///////////////////////////////////////////

                        gemv(mat_b, my_vec, buf, N);
                        gemv(mat_a, buf, gemv_out1, N);
                        gemv(mat_c, my_vec, gemv_out2, N);
                        if (!is_same_vec(gemv_out1, gemv_out2, N)) {
                                cerr << "Freivalds' Algorithm Failed" << endl;
                                return -1;
                        }
                }
                auto end = std::chrono::steady_clock::now();
                diff = end - start;
                std::cout << "Freivalds' Algorithm took " << diff.count() << " sec"
                                  << std::endl;
                cout << "Freivalds' Algorithm Passed" << endl;
        }

        return 0;
}
             