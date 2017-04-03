cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH "./CMake/;${CMAKE_MODULE_PATH}")

# Project version
set(WORDRANK_VERSION_MAJOR 1)
set(WORDRANK_VERSION_MINOR 0)

FIND_PACKAGE(Boost 1.40 COMPONENTS program_options REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
set(CMAKE_BUILD_TYPE Release)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
link_libraries(${MPI_LIBRARIES})

find_package(OpenMP REQUIRED)

find_package(MKL)
if (MKL_FOUND)
  include_directories(${MKL_INCLUDE_DIR})
  link_libraries(${MKL_LIBRARIES})
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -std=c++11 -Wall -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -O3 -Wall -fopenmp")

if (${CMAKE_CXX_COMPILER} MATCHES "icpc.*$")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xhost")
endif()

if (MKL_FOUND)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D USE_MKL")
endif()

ADD_EXECUTABLE(wordrank wordrank.cpp)

TARGET_LINK_LIBRARIES(wordrank ${Boost_LIBRARIES})

