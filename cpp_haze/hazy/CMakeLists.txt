set(CMAKE_CXX_COMPILER /usr/bin/g++-9)

cmake_minimum_required(VERSION 3.10)
project(hazy)
set(CMAKE_CXX_STANDARD 14)
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})
add_executable(hazy  main.cpp Image.h ImageConverter.h KDTree.h AccumMap.h Optimizer.cpp Optimizer.h)
target_link_libraries(hazy   ${PNG_LIBRARY})
