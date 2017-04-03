/*
 * Copyright (c) 2015 Shihao Ji and Hyokun Yun. All Rights Reserved.
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 * For more information, bug reports, fixes, contact:
 *   Shihao Ji  (shihaoji@yahoo.com)
 *   Hyokun Yun (yungilbert@gmail.com)
 */
#ifndef __WORDRANK_MODEL_HPP
#define __WORDRANK_MODEL_HPP

#include <queue>
#include <map>
#include <unordered_set>
#include <set>
#include <sstream>
#include <omp.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <mm_malloc.h>
#include "parameter.hpp"
#include "data.hpp"

#ifdef USE_MKL
#include "mkl.h"
#endif

using namespace boost::posix_time;
using std::cout;
using std::flush;
using std::endl;
using std::pair;
using std::priority_queue;
using std::map;
using std::set;
using std::min;
using std::max;
using std::max_element;

namespace wordrank {

struct aux_info {
    int col_index;
    int row_index;
    scalar xi;
    scalar wi;
};

scalar logistic_loss(scalar x) {
    return log2f(1.f + powf(2.f, x));
}

scalar hinge_loss(scalar x) {
    return x > -1.f ? x + 1.f : 0.f;
}

scalar logistic_der(scalar x) {
    return 1.f / (1.f + powf(2.f, -x));
}

scalar hinge_der(scalar x) {
    return x > -1.f ? 1.f : 0.f;
}

class Model {

private:
    Parameter& param_;
    Data& data_;
    rng_type rng_;

public:
    Model(Parameter& param, Data& data) : param_(param), data_(data), rng_(param.random_seed_) {}

    ~Model() {}

    void run();

private:
    scalar calc_loss(scalar x) {
        if (param_.loss_type_ == LossType::LOGISTIC) {
            return logistic_loss(x);
        } else if (param_.loss_type_ == LossType::HINGE) {
            return hinge_loss(x);
        } else {
            //cerr << "unsupported loss type" << endl;
            return 1000000.f;
            //exit(1);
        }
    }

    scalar calc_der(scalar x) {
        if (param_.loss_type_ == LossType::LOGISTIC) {
            return logistic_der(x);
        } else if (param_.loss_type_ == LossType::HINGE) {
            return hinge_der(x);
        } else {
            //cerr << "unsupported loss type" << endl;
            return 1000000.f;
            //exit(1);
        }
    }

    void normalize(scalar* matrix, int m, int n) {
        for (int i = 0; i < m; i++) {
            int offset = i * n;
            scalar norm2 = 0.f;
            #pragma simd
            for (int j = 0; j < n; j++) {
                scalar v = matrix[offset + j];
                norm2 += v * v;
            }
            scalar norm2r = 1.f / sqrtf(norm2);
            if (norm2 > 1e-6f) {
                #pragma simd
                for (int j = 0; j < n; j++) matrix[offset + j] *= norm2r;
            }
        }
    }

};

}

void writelog(const char *str) {
    cout << "\n" << second_clock::local_time() << " - " << str << endl << flush;
}

void wordrank::Model::run() {

    writelog("start run()");

    // define necessary constants
    const int rank = param_.rank_;
    const int dim = param_.latent_dim_;
    const scalar alpha = param_.alpha_;
    const scalar beta = param_.beta_;
    const scalar tl = param_.t_;
    const scalar tau = param_.tau_;
    const int num_rows = data_.get_num_rows();
    const int num_cols = data_.get_num_cols();

    const int numtasks = param_.numtasks_;
    const int numthreads = param_.numthreads_;
    const int numparts = numthreads * numtasks;
    const int numrows_per_part = data_.numrows_per_part_;
    const int numcols_per_part = data_.numcols_per_part_;
    const int num_rows_upbd = numthreads * numrows_per_part * numtasks;
    const int num_cols_upbd = numthreads * numcols_per_part * numtasks;

    const int sgd_num = param_.sgd_num_;
    const TransformType trans_type = param_.trans_type_;

    auto& train_row_nnzs = data_.train_row_nnzs_;
    auto& train_col_nnzsum = data_.train_col_nnzsum_;
    auto& train_row_nnzsum = data_.train_row_nnzsum_;

    const size_t train_num_points = data_.train_total_nnz_;
    const double train_total_nnzsum = data_.train_total_nnzsum_;

    vector<vector<record_info>>& csc_indices = data_.csc_indices_;
    vector<vector<index_type>>& csc_ptrs = data_.csc_ptrs_;
    vector<index_type>& local_nnz = data_.local_nnz_;

    // create thread-specific RNGs
    rng_type rngs[numthreads];
    for (int i = 0; i < numthreads; i++) {
        rngs[i] = rng_type(param_.random_seed_ + 15791 * i + 373 * rank);
    }
    rng_type main_rng(param_.random_seed_ + 377 * rank);

    /////////////////////////////////////////////////////////////////////
    // Initialize Parameters
    /////////////////////////////////////////////////////////////////////

#ifdef USE_MKL
    mkl_set_num_threads(1);
#endif

    // indexed by thread, local_row_index, coordinate

    scalar *matrix_U = (scalar *) _mm_malloc(numthreads * numrows_per_part * dim * sizeof(scalar), 64);

    scalar *matrix_V = (scalar *) _mm_malloc(numthreads * numcols_per_part * dim * sizeof(scalar), 64);

    scalar *vector_Rec = (scalar *) _mm_malloc(numthreads * num_cols_upbd * sizeof(scalar), 64);

    // maps local_col_index to global_col_index
    int *col_indices = (int *) _mm_malloc(numthreads * numcols_per_part * sizeof(int), 64);

    {
        std::uniform_real_distribution<scalar> init_dist(0, 1.f);

        #pragma omp parallel for num_threads(numthreads)
        for (int thread_index = 0; thread_index < numthreads; thread_index++)
        {
            scalar *local_matrix_U = matrix_U + thread_index * numrows_per_part * dim;
            for (int j = 0; j < numrows_per_part * dim; j++) {
                local_matrix_U[j] = init_dist(rngs[thread_index]);
            }
            scalar *local_matrix_V = matrix_V + thread_index * numcols_per_part * dim;
            for (int j = 0; j < numcols_per_part * dim; j++) {
                local_matrix_V[j] = init_dist(rngs[thread_index]);
            }
            // normalize them
            normalize(local_matrix_U, numrows_per_part, dim);
            normalize(local_matrix_V, numcols_per_part, dim);

            int *local_col_indices = col_indices + thread_index * numcols_per_part;
            #pragma simd
            for (int j = 0; j < numcols_per_part; j++) {
                int col_index = (rank * numthreads + thread_index) * numcols_per_part + j;
                if (col_index < num_cols) {
                    local_col_indices[j] = col_index;
                } else {
                    local_col_indices[j] = -1;
                }
            }
        }
        // set the values of overflowed rows and cols to zeros
        int numRowsOverflow = (rank + 1) * numthreads * numrows_per_part - num_rows;
        if (numRowsOverflow > 0)
            memset(matrix_U + numthreads * numrows_per_part - numRowsOverflow, 0, numRowsOverflow * sizeof(scalar));

        int numColsOverflow = (rank + 1) * numthreads * numcols_per_part - num_cols;
        if (numColsOverflow > 0)
            memset(matrix_V + numthreads * numcols_per_part - numColsOverflow, 0, numColsOverflow * sizeof(scalar));

    }

    scalar *global_matrix_U = NULL;
    if (rank == 0) {
        global_matrix_U = (scalar *) _mm_malloc(num_rows_upbd * dim * sizeof(scalar), 64);
    }

    scalar *global_matrix_V = (scalar *) _mm_malloc(num_cols_upbd * dim * sizeof(scalar), 64);

    int *global_col_indices = (int *) _mm_malloc(num_cols_upbd * sizeof(int), 64);

    MPI_Allgather(matrix_V, numthreads * numcols_per_part * dim, MPI_SCALAR, global_matrix_V,
            numthreads * numcols_per_part * dim, MPI_SCALAR, MPI_COMM_WORLD);
    MPI_Allgather(col_indices, numthreads * numcols_per_part, MPI_INT, global_col_indices,
            numthreads * numcols_per_part, MPI_INT, MPI_COMM_WORLD);

    /////////////////////////////////////////////////////////////////////
    // Optimization
    /////////////////////////////////////////////////////////////////////

    int *col_locations = (int *) _mm_malloc(num_cols * sizeof(int), 64);

    // prepare auxiliary information variables
    vector<vector<aux_info>> auxs(numthreads);
    vector<vector<vector<aux_info *>>> rowwise_auxptrs(numthreads, vector<vector<aux_info *>>(numrows_per_part, vector<aux_info *>()));

    #pragma omp parallel for num_threads(numthreads)
    for (int thread_index = 0; thread_index < numthreads; thread_index++) {
        auxs[thread_index].resize(local_nnz[thread_index]);

        aux_info *ptr = &auxs[thread_index][0];
        for (int col_index = 0; col_index < num_cols; col_index++) {
            for (int j = csc_ptrs[thread_index][col_index]; j < csc_ptrs[thread_index][col_index + 1]; j++) {
                record_info& ri = csc_indices[thread_index][j];
                ptr->col_index = col_index;
                ptr->row_index = ri.row_index;
                ptr->xi = (trans_type == TransformType::RHO2) ? 0.5f : 1.0f;
                ptr->wi = ri.weight;
                rowwise_auxptrs[thread_index][ri.row_index].push_back(ptr);
                ptr++;
            }
        }
    }
    // release memory
    csc_indices.clear();

    const double stepsize = param_.learning_rate_ * train_num_points / train_total_nnzsum;
    const double reg = param_.regularization_ / ((num_cols - 1) * train_num_points);

    writelog("starts optimization");

    int *global_perm = (int *) _mm_malloc(num_cols_upbd * sizeof(int), 64);
    int *local_perm = (int *) _mm_malloc(numthreads * numcols_per_part * sizeof(int), 64);

    double cumul_computation_time = 0.0;

    /*************************************************************************/
    /* Here is the main loop                                                 */
    /*************************************************************************/
    for (int iter_num = 0; iter_num < param_.max_iteration_; iter_num++) {

        std::stringstream monitor_stream;
        monitor_stream << iter_num << ", " << param_;

        cout << "iteration: " << iter_num << endl << flush;

        if ((param_.dump_prefix_.length() > 0) && (iter_num % param_.dump_period_ == 0)) {

            cout << "dumping data" << endl << flush;

            if (rank == 0) {

                MPI_Gather(matrix_U, numthreads * numrows_per_part * dim, MPI_SCALAR, global_matrix_U,
                        numthreads * numrows_per_part * dim, MPI_SCALAR, 0, MPI_COMM_WORLD);

                std::ofstream ofile(param_.dump_prefix_ + "_word_" + boost::lexical_cast<std::string>(iter_num) + ".txt");

                scalar *user_param = global_matrix_U;
                for (int user_index = 0; user_index < num_rows; user_index++) {
                    int global_row_index = data_.row_perm_inv_[user_index];
                    if (data_.useVocab_){
                        ofile << data_.vocab_[global_row_index];
                    } else {
                        ofile << global_row_index + 1;
                    }

                    for (int i = 0; i < dim; i++) {
                        ofile << " ";
                        ofile << user_param[i];
                    }
                    ofile << endl;
                    user_param += dim;
                }
                ofile.close();
            } else {
                MPI_Gather(matrix_U, numthreads * numrows_per_part * dim, MPI_SCALAR, global_matrix_U,
                        numthreads * numrows_per_part * dim, MPI_SCALAR, 0, MPI_COMM_WORLD);
            }

            if (rank == 0) {
                std::ofstream ofile(param_.dump_prefix_ + "_context_" + boost::lexical_cast<std::string>(iter_num) + ".txt");
                scalar *item_param = global_matrix_V;
                for (int item_index = 0; item_index < num_cols_upbd; item_index++) {
                    int global_col_index = global_col_indices[item_index];

                    if (global_col_index > -1) {
                        if (data_.useVocab_) {
                            ofile << data_.vocab_[global_col_index];
                        } else {
                            ofile << global_col_index + 1;
                        }

                        for (int i = 0; i < dim; i++) {
                            ofile << " ";
                            ofile << item_param[i];
                        }
                        ofile << endl;
                    }
                    item_param += dim;
                }
                ofile.close();
            }
        }


        double compute_start_time = omp_get_wtime();

        // sample a new assignment of column indices
        if (rank == 0) {
            std::iota(global_perm, global_perm + num_cols_upbd, 0);
            std::shuffle(global_perm, global_perm + num_cols_upbd, main_rng);
        }
        MPI_Scatter(global_perm, numcols_per_part * numthreads, MPI_INT, local_perm, numcols_per_part * numthreads,
                MPI_INT, 0, MPI_COMM_WORLD);

        // copy global parameters to local space
        {
            #pragma omp parallel for num_threads(numthreads)
            for (int i = 0; i < numcols_per_part * numthreads; i++) {
                // copy the index
                col_indices[i] = global_col_indices[local_perm[i]];

                // copy the parameter value
                scalar *source_ptr = global_matrix_V + local_perm[i] * dim;
                scalar *target_ptr = matrix_V + i * dim;
                std::copy(source_ptr, source_ptr + dim, target_ptr);
            }
        }

        // inside a local machine, we use a permutation matrix
        // to determine the order of block assignment
        vector<vector<int> > perm_matrix(numthreads, vector<int>(numthreads, 0));
        {
            vector<int> col_perm(numthreads, 0);

            #pragma omp parallel for num_threads(numthreads)
            for (int i = 0; i < numthreads; i++) {
                col_perm[i] = i;
                #pragma simd
                for (int j = 0; j < numthreads; j++) {
                    perm_matrix[i][j] = (i + j) % numthreads;
                }
            }
            shuffle(perm_matrix.begin(), perm_matrix.end(), main_rng);
            shuffle(col_perm.begin(), col_perm.end(), main_rng);

            // apply column permutation
            #pragma omp parallel for num_threads(numthreads)
            for (int i = 0; i < numthreads; i++) {
                vector<int> tmp_buffer(numthreads, 0);
                std::copy(perm_matrix[i].begin(), perm_matrix[i].end(), tmp_buffer.begin());
                #pragma simd
                for (int j = 0; j < numthreads; j++) {
                    perm_matrix[i][col_perm[j]] = tmp_buffer[j];
                }
            }
        }

        // run SGD
        for (int inner_iter = 0; inner_iter < numthreads; inner_iter++) {

            #pragma omp parallel for num_threads(numthreads)
            for (int thread_index = 0; thread_index < numthreads; thread_index++)
            {
                int chunk_index = perm_matrix[inner_iter][thread_index];

                int *local_col_indices = col_indices + chunk_index * numcols_per_part;
                scalar *local_matrix_U = matrix_U + thread_index * numrows_per_part * dim;
                scalar *local_matrix_V = matrix_V + chunk_index * numcols_per_part * dim;

                std::uniform_int_distribution<int> init_dist(0, numcols_per_part - 1);
                auto sample_column = [&](int local_col_index)->int {
                    int ret = local_col_index;
                    while (true) {
                        ret = init_dist(rngs[thread_index]);
                        if ((ret != local_col_index) && (local_col_indices[ret] != -1)) {
                            return ret;
                        }
                    }
                };

                scalar backup_row_vec[dim];

                if (param_.loss_type_ == LossType::LOGISTIC) {

                    for (int sgd_iter = 0; sgd_iter < sgd_num; sgd_iter++) {

                        for (int i = 0; i < numcols_per_part; i++) {
                            int col_index = local_col_indices[i];
                            scalar *col_vec = local_matrix_V + i * dim;

                            if (col_index < 0) {
                                continue;
                            }

                            for (int j = csc_ptrs[thread_index][col_index]; j < csc_ptrs[thread_index][col_index + 1]; j++) {
                                aux_info& aux_ptr = auxs[thread_index][j];
                                int row_index = aux_ptr.row_index;
                                scalar xi = aux_ptr.xi;
                                scalar wi = aux_ptr.wi;
                                scalar *row_vec = local_matrix_U + row_index * dim;

                                int sam_local_index = sample_column(i);
                                int sam_index = local_col_indices[sam_local_index];
                                scalar *sam_vec = local_matrix_V + sam_local_index * dim;

                                std::copy(row_vec, row_vec + dim, backup_row_vec);

                                scalar org_dot = std::inner_product(row_vec, row_vec + dim, col_vec, 0.f);
                                scalar sam_dot = std::inner_product(row_vec, row_vec + dim, sam_vec, 0.f);

                                scalar coef = 0.f;

                                switch (trans_type) {
                                case TransformType::RHO1:
                                    coef = wi * xi * INV_LOG2 / (tau + powf(2.f, org_dot - sam_dot));
                                    break;
                                case TransformType::RHO2: {
                                    scalar logxi = logf(xi);
                                    scalar logxi2 = logxi * logxi;
                                    coef = wi * xi * LOG2 / (logxi2 * (tau + powf(2.f, org_dot - sam_dot)));
                                    break;
                                }
                                case TransformType::RHO3:
                                    coef = wi * powf(xi, tl) / (tau + powf(2.f, org_dot - sam_dot));
                                    break;
                                default:
                                    cerr << "unsupported transform type" << endl;
                                    exit(1);
                                }

                                #pragma simd
                                for (int t = 0; t < dim; t++) {
                                    row_vec[t] -= stepsize * (reg * train_row_nnzsum[thread_index][row_index] * row_vec[t]
                                                     + coef * (sam_vec[t] - col_vec[t]));
                                    col_vec[t] -= stepsize * (reg * train_col_nnzsum[col_index] * col_vec[t]
                                                     - coef * backup_row_vec[t]);
                                    sam_vec[t] -= stepsize * (reg * (train_total_nnzsum - train_col_nnzsum[sam_index]) * sam_vec[t]
                                                     + coef * backup_row_vec[t]);
                                }
                            }
                        }
                    } // end of sam_iter
                } // case of logistic loss (end)
                else if (param_.loss_type_ == LossType::HINGE) {

                    for (int sgd_iter = 0; sgd_iter < sgd_num; sgd_iter++) {

                        for (int i = 0; i < numcols_per_part; i++) {
                            int col_index = local_col_indices[i];
                            scalar *col_vec = local_matrix_V + i * dim;

                            if (col_index < 0) {
                                continue;
                            }

                            for (int j = csc_ptrs[thread_index][col_index]; j < csc_ptrs[thread_index][col_index + 1]; j++) {
                                aux_info& aux_ptr = auxs[thread_index][j];
                                int row_index = aux_ptr.row_index;
                                scalar xi = aux_ptr.xi;
                                scalar wi = aux_ptr.wi;
                                scalar *row_vec = local_matrix_U + row_index * dim;

                                int sam_local_index = sample_column(i);
                                int sam_index = local_col_indices[sam_local_index];
                                scalar *sam_vec = local_matrix_V + sam_local_index * dim;

                                std::copy(row_vec, row_vec + dim, backup_row_vec);

                                scalar org_dot = std::inner_product(row_vec, row_vec + dim, col_vec, 0.f);
                                scalar sam_dot = std::inner_product(row_vec, row_vec + dim, sam_vec, 0.f);

                                // if hinge loss is nonzero
                                if (sam_dot > org_dot - tau) {

                                    scalar coef = 0.f;

                                    switch (trans_type) {
                                    case TransformType::RHO1:
                                        coef = wi * xi * INV_LOG2;
                                        break;
                                    case TransformType::RHO2: {
                                        scalar logxi = logf(xi);
                                        scalar logxi2 = logxi * logxi;
                                        coef = wi * xi * LOG2 / logxi2;
                                        break;
                                    }
                                    case TransformType::RHO3:
                                        coef = wi * powf(xi, tl);
                                        break;
                                    default:
                                        cerr << "unsupported transform type" << endl;
                                        exit(1);
                                    }

                                    #pragma simd
                                    for (int t = 0; t < dim; t++) {
                                        row_vec[t] -= stepsize * (reg * train_row_nnzsum[thread_index][row_index] * row_vec[t]
                                                        + coef * (sam_vec[t] - col_vec[t]));
                                        col_vec[t] -= stepsize * (reg * train_col_nnzsum[col_index] * col_vec[t]
                                                        - coef * backup_row_vec[t]);
                                        sam_vec[t] -= stepsize * (reg * (train_total_nnzsum - train_col_nnzsum[sam_index]) * sam_vec[t]
                                                        + coef * backup_row_vec[t]);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    cerr << "unsupported loss type 1" << endl << flush;
                    exit(1);
                }
            }
        }

        // at the end of the iteration, synchronize parameters
        MPI_Allgather(matrix_V, numthreads * numcols_per_part * dim, MPI_SCALAR, global_matrix_V,
                numthreads * numcols_per_part * dim, MPI_SCALAR, MPI_COMM_WORLD);
        MPI_Allgather(col_indices, numthreads * numcols_per_part, MPI_INT, global_col_indices,
                numthreads * numcols_per_part, MPI_INT, MPI_COMM_WORLD);

        #pragma omp parallel for num_threads(numthreads)
        #pragma simd
        for (int i = 0; i < num_cols_upbd; i++) {
            if (global_col_indices[i] >= 0) {
                col_locations[global_col_indices[i]] = i;
            }
        }

        // update auxiliary parameters exactly
        if (iter_num > 0 && iter_num % param_.xi_period_ == 0) {

            writelog("update auxiliary parameters");

            vector<scalar> max_xi(numthreads, -1000000.f);
            vector<scalar> min_xi(numthreads, 1000000.f);
            vector<scalar> mean_xi(numthreads, 0.f);


//            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, numthreads * numrows_per_part, num_cols_upbd,
//                    dim, 1.0f, matrix_U, dim, global_matrix_V, dim, 0.0f, matrix_Rec, num_cols_upbd);

            #pragma omp parallel for num_threads(numthreads)
            for (int thread_index = 0; thread_index < numthreads; thread_index++)
            {
                scalar* vector_Rec_ptr = vector_Rec + thread_index * num_cols_upbd;
                scalar* matrix_U_ptr = matrix_U + thread_index * numrows_per_part * dim;

                for (int row_index = 0; row_index < numrows_per_part; row_index++) {
#ifndef USE_MKL
                    scalar* row_ptr = matrix_U_ptr + row_index * dim;
                    for (int i = 0; i < num_cols_upbd; i++) {
                        scalar* col_ptr = global_matrix_V + i * dim;
                        scalar sum = 0.f;
                        #pragma simd
                        for (int j = 0; j < dim; j++) {
                            sum += row_ptr[j] * col_ptr[j];
                        }
                        vector_Rec_ptr[i] = sum;
                    }
#else
                    cblas_sgemv(CblasRowMajor, CblasNoTrans, num_cols_upbd, dim, 1.0f, global_matrix_V, dim,
                            matrix_U_ptr + row_index * dim, 1, 0.0f, vector_Rec_ptr, 1);
#endif
                    vector<aux_info *>& auxptrs = rowwise_auxptrs[thread_index][row_index];
                    int numnnzs = auxptrs.size();

                    for (int i = 0; i < numnnzs; i++) {
                        aux_info* ptr = auxptrs[i];
                        int col_index = ptr->col_index;
                        scalar wi = ptr->wi;
                        scalar val = vector_Rec_ptr[col_locations[col_index]];
                        double sum = 0;
                        #pragma simd
                        for (int j = 0; j < num_cols_upbd; j++) {
                            scalar diff = vector_Rec_ptr[j] - val;
                            sum += diff > -tau ? (tau + diff) : 0.;
                        }
                        ptr->xi = (trans_type == TransformType::RHO2) ? (0.5 * beta + 1) / (sqrt(sum) + 1 + beta) :
                                alpha / (sum + beta);
                    }

                    #pragma simd
                    for (int i = 0; i < numnnzs; i++) {
                        aux_info* ptr = auxptrs[i];
                        mean_xi[thread_index] += ptr->xi;
                        if (max_xi[thread_index] < ptr->xi)
                            max_xi[thread_index] = ptr->xi;
                        if (min_xi[thread_index] > ptr->xi)
                            min_xi[thread_index] = ptr->xi;
                    }

                }
            }

            scalar machine_max_xi = *max_element(max_xi.begin(), max_xi.end());
            scalar machine_min_xi = *min_element(min_xi.begin(), min_xi.end());
            scalar machine_mean_xi = std::accumulate(mean_xi.begin(), mean_xi.end(), 0.f) * numtasks / train_num_points;

            scalar global_max_xi = -1000000.f;
            scalar global_min_xi = 1000000.f;
            scalar global_mean_xi = 0.f;

            MPI_Allreduce(&machine_max_xi, &global_max_xi, 1, MPI_SCALAR, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(&machine_min_xi, &global_min_xi, 1, MPI_SCALAR, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&machine_mean_xi, &global_mean_xi, 1, MPI_SCALAR, MPI_SUM, MPI_COMM_WORLD);

            global_mean_xi /= numtasks;

            cout << "machine min xi: " << machine_min_xi << ", machine max xi: " << machine_max_xi
                    << ", machine mean xi: " << machine_mean_xi << ", global mean xi: " << global_mean_xi << endl << flush;
        }

        double computation_time = omp_get_wtime() - compute_start_time;
        cumul_computation_time += computation_time;
        monitor_stream << ", " << cumul_computation_time;

        // calculate the objective function
        if (param_.cost_eval_)
        {
            writelog("calculate cost function");

            vector<double> local_loss_sums(numthreads, 0);
            vector<double> local_approx_loss_sums(numthreads, 0);
            vector<double> local_reg_sums(numthreads, 0);

            #pragma omp parallel for num_threads(numthreads)
            for (int thread_index = 0; thread_index < numthreads; thread_index++)
            {
                scalar* matrix_U_ptr = matrix_U + thread_index * numrows_per_part * dim;
                scalar* vector_Rec_ptr = vector_Rec + thread_index * num_cols_upbd;

                for (int n = 0; n < local_nnz[thread_index]; n++) {
                    aux_info& aux = auxs[thread_index][n];
                    int row_index = aux.row_index;
                    int col_index = aux.col_index;
                    scalar xi = aux.xi;
                    scalar wi = aux.wi;

#ifndef USE_MKL
                    scalar* row_ptr = matrix_U_ptr + row_index * dim;
                    for (int i = 0; i < num_cols_upbd; i++) {
                        scalar* col_ptr = global_matrix_V + i * dim;
                        scalar sum = 0.f;
                        #pragma simd
                        for (int j = 0; j < dim; j++) {
                            sum += row_ptr[j] * col_ptr[j];
                        }
                        vector_Rec_ptr[i] = sum;
                    }
#else
                    cblas_sgemv(CblasRowMajor, CblasNoTrans, num_cols_upbd, dim, 1.f, global_matrix_V, dim,
                            matrix_U_ptr + row_index * dim, 1, 0.f, vector_Rec_ptr, 1);
#endif

                    scalar val = vector_Rec_ptr[col_locations[col_index]];
                    scalar sum = 0.f;
                    #pragma simd
                    for (int j = 0; j < num_cols_upbd; j++) {
                        scalar diff = vector_Rec_ptr[j] - val;
                        sum += diff > -tau ? (tau + diff) : 0.f;
                    }
                    if (trans_type == TransformType::RHO1) {
                        local_loss_sums[thread_index] += wi * log2(sum);
                        local_approx_loss_sums[thread_index] += wi * ((xi * (sum + beta) - 1) * INV_LOG2 - alpha * log2(xi));
                    } else {
                        scalar logxi = logf(xi);
                        scalar logxi2 = logxi * logxi;
                        local_loss_sums[thread_index] += wi * (1 - 1 / log2(sum + 1));
                        local_approx_loss_sums[thread_index] += wi * (1 + LOG2 * (logxi - 1 + xi * (sum + 1)) / logxi2
                                + (beta * xi - 0.5 * beta * logxi) * INV_LOG2);
                    }
                }

                // calculate the regularizer term
                if (param_.regularization_ > 0) {
                    #pragma simd
                    for (int row_index = 0; row_index < numrows_per_part; row_index++) {
                        scalar *row_vec = matrix_U_ptr + row_index * dim;
                        local_reg_sums[thread_index] += train_row_nnzsum[thread_index][row_index]
                                * std::inner_product(row_vec, row_vec + dim, row_vec, 0.f);
                    }
                    int *local_col_indices = col_indices + thread_index * numcols_per_part;
                    scalar *local_matrix_V = matrix_V + thread_index * numcols_per_part * dim;
                    #pragma simd
                    for (int col_index = 0; col_index < numcols_per_part; col_index++) {
                        scalar *col_vec = local_matrix_V + col_index * dim;
                        local_reg_sums[thread_index] += train_col_nnzsum[local_col_indices[col_index]]
                                * std::inner_product(col_vec, col_vec + dim, col_vec, 0.f);
                    }
                }
            }

            double global_loss_sum = 0.0;
            double machine_loss_sum = std::accumulate(local_loss_sums.begin(), local_loss_sums.end(), 0.0);
            MPI_Allreduce(&machine_loss_sum, &global_loss_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            double global_approx_loss_sum = 0.0;
            double machine_approx_loss_sum = std::accumulate(local_approx_loss_sums.begin(), local_approx_loss_sums.end(), 0.0);
            MPI_Allreduce(&machine_approx_loss_sum, &global_approx_loss_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            double global_reg_sum = 0.0;
            if (param_.regularization_ > 0) {
                double machine_reg_sum = std::accumulate(local_reg_sums.begin(), local_reg_sums.end(), 0.0);
                MPI_Allreduce(&machine_reg_sum, &global_reg_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            }

            double cost = global_approx_loss_sum + param_.regularization_ * global_reg_sum;

            monitor_stream << ", cost: " << cost << " (" << global_approx_loss_sum << " + " << param_.regularization_ * global_reg_sum
                    << "), true loss: " << global_loss_sum;
        }

        if (rank == 0) {
            cout << "monitor, " << monitor_stream.str() << endl << flush;
        }

    } // end of SGD iteration

    /////////////////////////////////////////////////////////////////////
    // Deallocate Memory
    /////////////////////////////////////////////////////////////////////

    _mm_free(vector_Rec);
    _mm_free(matrix_U);
    _mm_free(matrix_V);
    if (rank == 0) _mm_free(global_matrix_U);
    _mm_free(global_matrix_V);
    _mm_free(global_col_indices);
    _mm_free(local_perm);
    _mm_free(global_perm);
    _mm_free(col_indices);
    _mm_free(col_locations);
}

#endif
