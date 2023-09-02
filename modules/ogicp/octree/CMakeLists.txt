cmake_minimum_required(VERSION 3.10.0)
project(goctomap)

find_package(octomap REQUIRED)
include_directories(${OCTOMAP_INCLUDE_DIRS})

message(STATUS "OCTOMAP_INCLUDE_DIRS: ${OCTOMAP_INCLUDE_DIRS}")
message(STATUS "OCTOMAP_LIBRARIES: ${OCTOMAP_LIBRARIES}")

add_library(goctomap SHARED GaussionOcTree.cc)
target_link_libraries(goctomap PUBLIC ${OCTOMAP_LIBRARIES})
target_include_directories(
  goctomap
  PUBLIC include
         ${PCL_INCLUDE_DIRS}
         ${EIGEN3_INCLUDE_DIR}
         ${OCTOMAP_INCLUDE_DIRS}
  )

option(BUILD_EXAMPLE "Build example" OFF)
if(BUILD_EXAMPLE)
  find_package(PCL REQUIRED)
  add_executable(test_ot test/test_ot.cpp)
  target_link_libraries(test_ot goctomap ${PCL_LIBRARIES})
  target_include_directories(
    test_ot PUBLIC ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR}
    )
endif()
