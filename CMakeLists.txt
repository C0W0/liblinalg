cmake_minimum_required(VERSION 3.24)
project(linalg)

set(CMAKE_CXX_STANDARD 17)

add_library(linalg
        main.cpp
        Matrix.tpp
        Matrix.h)
add_executable(linalg_test main.cpp
        Matrix.tpp
        Matrix.h)
