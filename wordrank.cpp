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
#include <boost/format.hpp>
#include "parameter.hpp"
#include "data.hpp"
#include "model.hpp"

using std::cout;
using std::cerr;
using std::endl;

using wordrank::Parameter;
using wordrank::Data;
using wordrank::Model;

int main(int argc, char **argv) {

    int numtasks, rank, hostname_len;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    // retrieve MPI task info
    int mpi_thread_provided;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &mpi_thread_provided);
    if (mpi_thread_provided != MPI_THREAD_MULTIPLE) {
        cerr << "MPI multiple thread not provided!!! (" << mpi_thread_provided << " != " << MPI_THREAD_MULTIPLE << ")" << endl;
        exit(1);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Get_processor_name(hostname, &hostname_len);

    cout << boost::format("processor name: %s, number of tasks: %d, rank: %d") % hostname % numtasks % rank << endl;

    Parameter param;
    Data data;

    int ret_code;

    if ((ret_code = param.read_arguments(argc, argv)) != 0) {
        return ret_code;
    }

    param.numtasks_ = numtasks;
    param.rank_ = rank;

    if ((ret_code = load_data(data, param.data_path_, param.random_seed_, rank, numtasks, param.numthreads_,
            param.xmax_, param.epsilon_)) != 0) {
        return ret_code;
    }

    Model model(param, data);
    model.run();

    MPI_Finalize();
    return 0;
}
