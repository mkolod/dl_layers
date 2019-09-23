cmake_minimum_required(VERSION 3.12)
project(batchnorm)

set(CMAKE_CXX_STANDARD 17)

add_executable(batchnorm batchnorm.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(batchnorm PUBLIC OpenMP::OpenMP_CXX)
