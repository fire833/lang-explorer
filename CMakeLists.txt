cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# debug this crap
set(CMAKE_CXX_FLAGS "-g")

# set the project name
project(LangExplorerUtils VERSION 1.0)
link_directories(../../oss/taco/build/lib/)
include_directories(../../oss/taco/include/)

# add the executable
add_executable(runner runners/taco/taco_runner.cpp)
target_link_libraries(runner taco)
