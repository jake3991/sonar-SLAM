cmake_minimum_required(VERSION 2.8.3)
project(bruce_slam)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

find_package(catkin REQUIRED
  rospy
  tf
  #rti_dvl
  #sonar_oculus
  #bar30_depth
  message_generation
  sensor_msgs
  geometry_msgs
  pybind11_catkin
)
find_package(catkin REQUIRED)
find_package(libpointmatcher REQUIRED)
find_package(PCL REQUIRED)

catkin_package()
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${libpointmatcher_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

catkin_python_setup()

#add_subdirectory(src/${PROJECT_NAME}/cpp/pybind11)
pybind11_add_module(pcl src/${PROJECT_NAME}/cpp/pcl.cpp)
target_link_libraries(pcl ${libpointmatcher_LIBRARIES} ${PCL_LIBRARIES})
set_target_properties(pcl PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}
)

pybind11_add_module(cfar src/${PROJECT_NAME}/cpp/cfar.cpp)
set_target_properties(cfar PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
