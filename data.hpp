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
#ifndef __WORDRANK_DATA_HPP
#define __WORDRANK_DATA_HPP

#include <fstream>
#include <vector>
#include <set>
#include <boost/tokenizer.hpp>
#include "wordrank.hpp"

namespace wordrank {

using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using std::vector;
using std::set;
using std::make_pair;
using boost::lexical_cast;
using boost::tokenizer;
using boost::char_separator;

struct record_info {
    index_type row_index;
    scalar weight;
};

class Data {

public:
    Data() : numthreads_(0), numrows_per_part_(0) {}

    ~Data() {}

    index_type get_num_rows() {
        return num_rows_;
    }

    index_type get_num_cols() {
        return num_cols_;
    }

    int numthreads_;
    index_type num_rows_;
    index_type num_cols_;
    index_type numrows_per_part_;
    index_type numcols_per_part_;
    int row_start_index_;

    vector<int> row_perm_;
    vector<int> row_perm_inv_;

    vector<vector<index_type>> train_row_nnzs_;
    vector<vector<set<index_type>>> train_rowwise_;

    vector<vector<record_info>> csc_indices_;
    vector<vector<index_type>> csc_ptrs_;

    vector<index_type> local_nnz_;
    vector<vector<scalar>> train_row_nnzsum_;
    vector<scalar> train_col_nnzsum_;
    vector<scalar> train_col_nnzweight_;
    size_t train_total_nnz_;
    double train_total_nnzsum_;

    vector<string> vocab_;
    bool useVocab_;
};

int load_data(Data& data, const string path, const int seed, const int rank, const int numtasks, const int numthreads, const int xmax, const scalar epsilon) {

    // initialize random number generator; 
    // it is important that every machine is using the same rng to partition the data
    rng_type rng(seed);

    cout << "data path: " << path << endl;

    string train_filename;
    string vocab_filename = "";
    data.useVocab_ = false;
    unsigned int num_train, num_words;

    //////////////////////////////////////////////////////////
    // Read Metadata
    //////////////////////////////////////////////////////////

    // this scope reads meta file
    {
        char_separator<char> sep(" ");

        string metafile_path = path + "/meta";
        ifstream metafile(metafile_path.c_str());

        cout << "reading metadata file: " << metafile_path << endl;

        string line;

        if (false == metafile.is_open()) {
            cerr << "could not open: " << metafile_path << endl;
            return 1;
        }

        // get size of data
        {
            getline(metafile, line);

            tokenizer<char_separator<char> > tokens(line, sep);
            tokenizer<char_separator<char> >::iterator iter = tokens.begin();

            data.num_rows_ = lexical_cast<index_type>(*iter);
            cout << "number of rows: " << data.num_rows_ << endl;

            iter++;
            data.num_cols_ = lexical_cast<index_type>(*iter);
            cout << "number of columns: " << data.num_cols_ << endl;
        }

        // get information about training data
        {
            getline(metafile, line);

            tokenizer<char_separator<char> > tokens(line, sep);
            tokenizer<char_separator<char> >::iterator iter = tokens.begin();

            num_train = lexical_cast<unsigned int>(*iter);

            iter++;
            train_filename = *iter;

            cout << "train_filename: " << train_filename << ", number of points: " << num_train << endl;
        }

        // get information about vocabulary
        {
            getline(metafile, line);
            if (line != "") {

                tokenizer<char_separator<char> > tokens(line, sep);
                tokenizer<char_separator<char> >::iterator iter = tokens.begin();

                num_words = lexical_cast<unsigned int>(*iter);

                iter++;
                vocab_filename = *iter;

                cout << "vocab_filename: " << vocab_filename << ", size of vocabulary: " << num_words << endl;

                data.useVocab_ = true;
            }
        }

    } // metafile read done

    data.train_total_nnz_ = num_train;

    //////////////////////////////////////////////////////////
    // Partition Data
    //////////////////////////////////////////////////////////

    // sample a row index permutation
    data.row_perm_.resize(data.num_rows_);
    data.row_perm_inv_.resize(data.num_rows_);
    std::iota(data.row_perm_.begin(), data.row_perm_.end(), 0);
    std::shuffle(data.row_perm_.begin(), data.row_perm_.end(), rng);

    #pragma omp parallel for num_threads(numthreads)
    #pragma simd
    for (int i = 0; i < data.num_rows_; i++) {
        data.row_perm_inv_[data.row_perm_[i]] = i;
    }

    const int numparts = numtasks * numthreads;
    const int numrows_per_part = data.num_rows_ / numparts + ((data.num_rows_ % numparts > 0) ? 1 : 0);
    const int numcols_per_part = data.num_cols_ / numparts + ((data.num_cols_ % numparts > 0) ? 1 : 0);
    data.numrows_per_part_ = numrows_per_part;
    data.numcols_per_part_ = numcols_per_part;

    data.numthreads_ = numthreads;

    int row_start_index = numrows_per_part * rank * numthreads;
    int row_end_index = std::min(numrows_per_part * (rank + 1) * numthreads, data.num_rows_);
    data.row_start_index_ = row_start_index;

    // allocate temporary data structure
    vector<vector<vector<record_info>>> colwise_vecs(numthreads);
    #pragma omp parallel for num_threads(numthreads)
    for (int i = 0; i < numthreads; i++) {
        colwise_vecs[i].resize(data.num_cols_);
    }

    //////////////////////////////////////////////////////////
    // Allocate Memories
    //////////////////////////////////////////////////////////
    {
        data.train_col_nnzsum_.resize(data.num_cols_, 0);
        data.train_row_nnzs_.resize(numthreads);
        data.train_row_nnzsum_.resize(numthreads);
        data.train_rowwise_.resize(numthreads);

        #pragma omp parallel for num_threads(numthreads)
        for (int i = 0; i < numthreads; i++) {
            data.train_row_nnzs_[i].resize(numrows_per_part, 0);
            data.train_row_nnzsum_[i].resize(numrows_per_part, 0);
            data.train_rowwise_[i].resize(numrows_per_part);
        }
    }

    //////////////////////////////////////////////////////////
    // Read Training Data
    //////////////////////////////////////////////////////////
    {
        data.local_nnz_.resize(numthreads, 0);

        string train_path = path + "/" + train_filename;

        ifstream file(train_path.c_str());

        if (false == file.is_open()) {
            cerr << "could not open: " << train_path << endl;
            return 1;
        }

        boost::char_separator<char> sep(" ");
        string line;
        unsigned int count = 0;

        while (file.good()) {

            if (count % 1000000 == 0) {
                cout << "reading train: " << count << " / " << num_train << " ("
                        << (static_cast<double>(count) / num_train * 100) << "%)" << endl;
            }

            getline(file, line);

            tokenizer<char_separator<char> > tokens(line, sep);
            tokenizer<char_separator<char> >::iterator iter = tokens.begin();

            if (iter == tokens.end()) {
                break;
            }

            // all indicies are subtracted by 1 to make it 0-based index
            int row_index = lexical_cast<int>(*iter) - 1;
            ++iter;

            if (iter == tokens.end()) {
                break;
            }

            int col_index = lexical_cast<int>(*iter) - 1;
            ++iter;

            if (iter == tokens.end()) {
                break;
            }

            scalar weight = lexical_cast<scalar>(*iter);
            weight = (weight < xmax) ? powf(weight / xmax, epsilon) : 1.f;
            ++iter;

            int perm_row_index = data.row_perm_[row_index];

            if (perm_row_index >= row_start_index && perm_row_index < row_end_index) {
                int thread_index = (perm_row_index - row_start_index) / numrows_per_part;
                int local_row_index = (perm_row_index - row_start_index) % numrows_per_part;
                record_info ri;
                ri.row_index = local_row_index;
                ri.weight = weight;
                colwise_vecs[thread_index][col_index].push_back(ri);
                data.train_rowwise_[thread_index][local_row_index].insert(col_index);

                data.train_row_nnzs_[thread_index][local_row_index]++;
                data.train_row_nnzsum_[thread_index][local_row_index] += weight;

                data.local_nnz_[thread_index]++;
            }

            data.train_col_nnzsum_[col_index] += weight;

            count++;

        }
        data.train_total_nnzsum_ = std::accumulate(data.train_col_nnzsum_.begin(), data.train_col_nnzsum_.end(), 0.0);

    }

    //////////////////////////////////////////////////////////
    // Make Training Data Compact
    //////////////////////////////////////////////////////////

    cout << "rank: " << rank << ", compact data" << endl;

    data.csc_ptrs_.resize(numthreads, vector<index_type>(data.num_cols_ + 1, 0));
    data.csc_indices_.resize(numthreads, vector<record_info>());

    #pragma omp parallel for num_threads(numthreads)
    for (int thread_index = 0; thread_index < numthreads; thread_index++) {
        data.csc_indices_[thread_index].reserve(data.local_nnz_[thread_index]);
        int current_pos = 0;
        for (int col_index = 0; col_index < data.num_cols_; col_index++) {
            data.csc_ptrs_[thread_index][col_index] = current_pos;
            for (record_info& ri : colwise_vecs[thread_index][col_index]) {
                data.csc_indices_[thread_index].push_back(ri);
                current_pos++;
            }
        }
        data.csc_ptrs_[thread_index][data.num_cols_] = current_pos;
    }


    //////////////////////////////////////////////////////////
    // Read vocabulary
    //////////////////////////////////////////////////////////
    if (rank == 0 && vocab_filename != "")
    {
        string vocab_path = path + "/" + vocab_filename;

        ifstream file(vocab_path.c_str());

        if (false == file.is_open()) {
            cerr << "could not open: " << vocab_path << endl;
            return 1;
        }

        string line;
        int count = 0;

        data.vocab_.clear();
        while (file.good()) {

            if (count % 100000 == 0) {
                cout << "reading vocabulary: " << count << " / " << num_words << " ("
                        << (static_cast<double>(count) / num_words * 100) << "%)" << endl;
            }

            getline(file, line);

            data.vocab_.push_back(line);

            count++;

            if (count >= num_words) break;

        }
        if (count != num_words) {
            cout << "size of vocabulary is smaller than " << num_words << endl;
            return 1;
        }

    }

    return 0;

}

}

#endif
