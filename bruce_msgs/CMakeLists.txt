cmake_minimum_required(VERSION 2.8.3)
project(bruce_msgs)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  message_generation
  sensor_msgs
  #sonar_oculus
)

add_message_files(
   FILES
   ISAM2Update.msg
   #keyframe_image.msg
   #posehistory.msg
)

add_service_files(
  FILES
  QueryExplorationPath.srv
  PredictSLAMUpdate.srv
  GetOccupancyMap.srv
)

generate_messages(
   DEPENDENCIES
   geometry_msgs
   nav_msgs
   sensor_msgs
   #sonar_oculus
)

#find_package(GTSAM REQUIRED)
#set(GTSAM_LIBRARIES gtsam)

#include_directories(
#  include
#  ${GTSAM_INCLUDE_DIR}
#  ${catkin_INCLUDE_DIRS}
#)

catkin_package(
   INCLUDE_DIRS include
   LIBRARIES conversions
   CATKIN_DEPENDS message_runtime geometry_msgs nav_msgs sensor_msgs
   #DEPENDS GTSAM
)

#add_library(conversions src/bruce_msgs/conversions.cpp)
#add_dependencies(conversions bruce_msgs_generate_messages_cpp)
#target_link_libraries(conversions  ${catkin_LIBRARIES})
