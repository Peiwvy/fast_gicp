add_subdirectory(octree)
add_library(ogicp_registration SHARED ogicp.cc ogicp/fast_ogicp.cpp
                                      ogicp/lsq_registration_depth.cpp
  )
target_include_directories(
  ogicp_registration PUBLIC ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR}
                            ${CMAKE_CURRENT_SOURCE_DIR}
  )
if(OPENMP_FOUND)
  if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(ogicp_registration OpenMP::OpenMP_CXX)
  endif()
endif()
target_link_libraries(ogicp_registration ${PCL_LIBRARIES} goctomap)
