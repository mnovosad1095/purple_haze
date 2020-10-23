project(haze_cpp)

set(CMAKE_CXX_COMPILER /usr/bin/g++-10)

cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})
add_executable(hazy  main.cpp Image.h ImageConverter.h KDTree.h Optimizer.cpp Optimizer.h Image.cpp)
target_link_libraries(hazy   ${PNG_LIBRARY})
