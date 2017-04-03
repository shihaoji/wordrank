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
#ifndef __WORDRANK_PARAMETER_HPP
#define __WORDRANK_PARAMETER_HPP

#include <string>
#include <vector>
#include <boost/program_options.hpp>
#include "wordrank.hpp"

namespace wordrank {

using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;

struct Parameter {

    string data_path_;
    double learning_rate_;
    double regularization_;
    int latent_dim_;
    int xmax_;
    scalar epsilon_;
    scalar alpha_;
    scalar beta_;
    scalar tau_;

    int max_iteration_;
    int random_seed_;
    int xi_period_;

    bool cost_eval_;

    TransformType trans_type_;
    scalar t_;
    LossType loss_type_;
    string trans_name_;
    string loss_name_;

    int numthreads_;
    int numtasks_;
    int rank_;

    string dump_prefix_;
    int dump_period_;

    int sgd_num_;

    int read_arguments(int argc, char **argv);

    friend std::ostream& operator<<(std::ostream& stream, const Parameter& param) {
        stream << param.latent_dim_ << ", " << param.regularization_ << ", "
                << param.learning_rate_ << ", " << param.xi_period_;
        return stream;
    }

};
}

int wordrank::Parameter::read_arguments(int argc, char **argv) {

    namespace po = boost::program_options;
    po::options_description desc("wordrank options:");
    desc.add_options()
            ("help", "produce help message")
            ("path", po::value<string>(&data_path_)->default_value("../data/wiki.toy"), "path of data")
            ("nthreads", po::value<int>(&numthreads_)->default_value(4), "number of threads for each MPI process")
            ("lrate", po::value<double>(&learning_rate_)->default_value(0.001), "value of learning rate parameter")
            ("reg", po::value<double>(&regularization_)->default_value(0.005), "value of regularization parameter")
            ("dim", po::value<int>(&latent_dim_)->default_value(100), "dimensionality of latent space")
            ("xmax", po::value<int>(&xmax_)->default_value(100), "cutoff value for weighting function")
            ("epsilon", po::value<scalar>(&epsilon_)->default_value(0.75), "power scaling value for weighting function")
            ("alpha", po::value<scalar>(&alpha_)->default_value(1), "alpha of the gamma distribution")
            ("beta", po::value<scalar>(&beta_)->default_value(0), "beta of the gamma distribution")
            ("tau", po::value<scalar>(&tau_)->default_value(1.0), "delta of logistic or hinge loss")
            ("iter", po::value<int>(&max_iteration_)->default_value(1000), "number of iterations")
            ("seed", po::value<int>(&random_seed_)->default_value(12345), "random number generator seed value")
            ("period", po::value<int>(&xi_period_)->default_value(100), "period of xi variable updates")
            ("cost_eval", po::bool_switch(&cost_eval_)->default_value(false), "evaluate the cost function")
            ("transform", po::value<string>(&trans_name_)->default_value("rho1"), "name of the transform (rho1, rho2, rho3)")
            ("t", po::value<scalar>(&t_)->default_value(1.5), "t value of log_t (rho3) transform function")
            ("loss", po::value<string>(&loss_name_)->default_value("logistic"), "name of the loss (logistic, hinge)")
            ("sgd_num", po::value<int>(&sgd_num_)->default_value(1), "number of SGD taken for each data point")
            ("dump_prefix", po::value<string>(&dump_prefix_)->default_value("model"), "prefix to the parameter dump file")
            ("dump_period", po::value<int>(&dump_period_)->default_value(100), "how often parameters should be dumped");

    bool flagHelp = false;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        flagHelp = true;
    }

    if (flagHelp == true) {
        cerr << desc << endl;
        return 1;
    }

    if (data_path_.length() <= 0) {
        cerr << "data file path not specified." << endl;
        return 1;
    }

    if (trans_name_ == "rho1") {
        trans_type_ = TransformType::RHO1;
    } else if (trans_name_ == "rho2") {
        trans_type_ = TransformType::RHO2;
    } else if (trans_name_ == "rho3") {
        trans_type_ = TransformType::RHO3;
    } else {
        cerr << "illegal transform name: " << trans_name_ << endl;
        return 1;
    }

    if (loss_name_ == "logistic") {
        loss_type_ = LossType::LOGISTIC;
    } else if (loss_name_ == "hinge") {
        loss_type_ = LossType::HINGE;
    } else {
        cerr << "illegal loss name: " << loss_name_ << endl;
        return 1;
    }

    return 0;

}

#endif
