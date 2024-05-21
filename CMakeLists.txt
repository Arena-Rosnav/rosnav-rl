cmake_minimum_required(VERSION 3.0.2)
project(rosnav)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  rospy
  tf2
  tf2_geometry_msgs
  tf2_ros
  message_generation
  dynamic_reconfigure
  nav_core
  mbf_costmap_core
  mbf_msgs
)

catkin_python_setup()

add_service_files(FILES
  GetAction.srv
)

generate_messages()

generate_dynamic_reconfigure_options(
  cfg/RosnavLocalPlannerReconfigure.cfg
)

catkin_package(
  INCLUDE_DIRS include cfg
  CATKIN_DEPENDS
  base_local_planner nav_core dynamic_reconfigure message_runtime costmap_2d costmap_converter pluginlib roscpp tf
)

include_directories(include)
include_directories(cfg)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${GSTREAMER_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/rosnav_local_planner.cpp
  src/rosnav_config.cpp
)

add_dependencies(rosnav ${PROJECT_NAME}_gencfg)
add_dependencies(rosnav ${${PROJECT_NAME}_EXPORTED_TARGETS})

target_link_libraries(rosnav ${catkin_LIBRARIES})

catkin_install_python(PROGRAMS
  scripts/rosnav_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

# # Mark cpp header files for installation
install(DIRECTORY include
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

# # Mark other files for installation (e.g. launch and bag files, etc.)
install(FILES
  rosnav_local_planner_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY
  cfg
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN ".svn" EXCLUDE
)
