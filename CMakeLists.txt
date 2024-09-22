cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
# debug this crap
set(CMAKE_CXX_FLAGS "-g")

# set the project name
project(TacoRunner VERSION 1.0)

# add the executable
add_executable(runner runners/taco/taco_runner.cpp)
link_directories(../../oss/taco/build/lib/)
include_directories(../../oss/taco/include/)
find_library(TACO_LIB taco)
