cmake_minimum_required(VERSION 3.8)
project(rosnav_rl)

# Find essential ROS2 components
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Service generation
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetAction.srv"
  DEPENDENCIES geometry_msgs nav_msgs
)

# Python installation setup
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/action_server.py
  DESTINATION lib/${PROJECT_NAME}
)

# Export dependencies
ament_export_dependencies(
  rclpy
  geometry_msgs
  nav_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  rosidl_default_runtime
)

ament_package()