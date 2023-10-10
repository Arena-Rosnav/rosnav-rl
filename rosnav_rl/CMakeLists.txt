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
)

catkin_python_setup()

add_service_files(FILES
  GetAction.srv
)

add_message_files(FILES
  ResetStackedObs.msg
)

generate_messages()

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/rosnav_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
