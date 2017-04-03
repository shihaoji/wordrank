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
#ifndef __WORDRANK_WORDRANK_HPP
#define __WORDRANK_WORDRANK_HPP

#include <iostream>
#include <random>
#include <mpi.h>

namespace wordrank {

#define MPI_SCALAR MPI_FLOAT
#define LOG2 0.6931471805599453f
#define INV_LOG2 1.4426950408889634f
//#define INV_SQRT2 0.7071067811865475
using scalar = float;
using index_type = int;

using rng_type = std::mt19937_64;

enum class LossType {
    LOGISTIC, HINGE
};

enum class TransformType {
    RHO1, RHO2, RHO3
};

}

#endif
