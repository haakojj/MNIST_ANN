cmake_minimum_required (VERSION 3.1)
project(MNIST)

set(EXE_NAME "MNIST")
set (CMAKE_CXX_STANDARD 11)

find_package(PNG REQUIRED)

set(EIGEN_INCLUDE_DIRS "eigen" CACHE FILEPATH "Path to Eigen.")
get_filename_component(EIGEN_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS} ABSOLUTE)

set(PNGPP_INCLUDE_DIRS "png++" CACHE FILEPATH "Path to png++.")
get_filename_component(PNGPP_INCLUDE_DIRS ${PNGPP_INCLUDE_DIRS} ABSOLUTE)

file(GLOB DATA_SETS "${CMAKE_SOURCE_DIR}/data/*")
message(STATUS ${DATA_SETS})

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
set(INSTALL_DIR "bin" CACHE FILEPATH "Installation path.")

add_subdirectory(src)
