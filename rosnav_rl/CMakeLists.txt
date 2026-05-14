cmake_minimum_required(VERSION 3.8)
project(rosnav_rl)

# Find essential ROS2 components
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pluginlib REQUIRED)
find_package(nav2_core REQUIRED)
find_package(rosnav_rl_msgs REQUIRED)  # Add dependency on your msgs package

# Include directories
include_directories(include)

add_library(rosnav_rl_planner SHARED
  src/drl_controller.cpp
)

target_include_directories(rosnav_rl_planner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(rosnav_rl_planner
  rclcpp
  geometry_msgs
  nav_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  nav2_core
  pluginlib
  rosnav_rl_msgs  # Add dependency
)

# Python installation - NOW WORKS without conflict
ament_python_install_package(rosnav_rl)

# Install everything else...
install(DIRECTORY include/ DESTINATION include/)
install(TARGETS rosnav_rl_planner DESTINATION lib)
install(PROGRAMS
  scripts/action_server.py
  scripts/arena_inference_node.py
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(FILES local_planner_plugin.xml DESTINATION share/${PROJECT_NAME})

# Export dependencies
ament_export_dependencies(
  rclpy
  geometry_msgs
  nav_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  nav2_core
  pluginlib
  rclcpp
  rosnav_rl_msgs  # Add dependency
)

ament_export_include_directories(include)
ament_export_libraries(rosnav_rl_planner)
pluginlib_export_plugin_description_file(nav2_core local_planner_plugin.xml)

ament_package()