#WordRank
WordRank is a word embedding algorithm that estimates vector representations for words via robust ranking. Similar to GloVe, WordRank's training is performed on aggregated word-word co-occurrence matrix from a corpus. But dissimilar to GloVe, where a regression loss is employed, WordRank optimizes a ranking-based loss. WordRank distributes computation across multiple machines via MPI to support large scale word embedding problems.

##License
All source code files in WordRank is under [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0), except `./CMake/FindMKL.cmake` which is adopted from https://github.com/Eyescale/CMake and is under BSD.

##Prerequisites
WordRank is developed and tested on UNIX-based systems, with the following software dependencies:

- C++ compiler with C++11 support ([Intel compiler](https://software.intel.com/en-us/qualify-for-free-software) is preferred; g++ is ok but all #pragma simd are ignored as of now, which lead to 2x-3x performance loss.)
- MPI library, with multi-threading support (Intel MPI, MPICH2 or MVAPICH2)
- OpenMP (No separated installation is needed once Intel compiler is installed)
- CMake (at least 2.6)
- Boost library (at least 1.49)
- [GloVe v.1.0](http://nlp.stanford.edu/projects/glove/) (for co-occurrence matrix preparation)
- [HyperWords](https://bitbucket.org/omerlevy/hyperwords) (for evaluation)
- MKL (optional)

##Environment Setup
* Install Intel Parallel Studio XE Cluster Edition (i.e., Intel compiler, OpenMP, MPI and MKL. [free copies](https://software.intel.com/en-us/qualify-for-free-software) are available for some users)
* Enable Intel C++ development environment
```
source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64 
source /opt/intel/impi_latest/bin64/mpivars.sh (pointing to the path of your installation)
```
* Install Boost library
```
sudo yum install boost-devel (on RedHat/Centos)
sudo apt-get install libboost-all-dev (on Ubuntu)
```
* [Intel compiler](https://software.intel.com/en-us/qualify-for-free-software) is preferred; g++ is ok but all #pragma simd are ignored as of now, which lead to 2x-3x performance loss.
 
##Quick Start
1. Download the code: ```git clone https://bitbucket.org/shihaoji/wordrank```
2. Run ```.\install.sh``` to build the package (e.g., it downloads GloVe v.1.0 and HyperWords and applies patches to them, and then compiles the source code. Intel compiler is used as default. See the switch in .\install.sh to use g++ instead.)
3. Run the demo script: ```cd scripts; ./demo.sh``` (NUM_CORES=16 by default, set this to # of physical cores of your machine)
4. Evaluate the models: ```cd scripts; ./eval.sh N (to evaluate the model after N iterations, e.g., N=200)```

##Reference
Shihao Ji, Hyokun Yun, Pinar Yanardag, Shin Matsushima, S. V. N. Vishwanathan. "[WordRank: Learning Word Embeddings via Robust Ranking](http://arxiv.org/abs/1506.02761)", Conference on Empirical Methods in Natural Language
Processing (EMNLP), Nov. 2016.