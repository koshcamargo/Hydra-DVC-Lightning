cmake_minimum_required(VERSION 3.12)
project(rowmean)

set(CMAKE_CXX_STANDARD 11)

find_package(pybind11 REQUIRED)

add_library(rowmean MODULE bindings.cpp)
target_link_libraries(rowmean PRIVATE pybind11::module)
