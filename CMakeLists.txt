cmake_minimum_required(VERSION 3.1)
project(sscan CXX)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(./Eigen)
add_executable(sscan sscan.cc)

set_property(TARGET sscan PROPERTY CXX_STANDARD 11)
