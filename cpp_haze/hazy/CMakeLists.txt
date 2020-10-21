cmake_minimum_required(VERSION 3.17)
project(hazy)

set(CMAKE_CXX_STANDARD 14)

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})
add_executable(hazy  main2.cpp Image.h ImageConverter.h KDTree.h AccumMap.h Optimizer.cpp Optimizer.h)
target_link_libraries(hazy   ${PNG_LIBRARY})