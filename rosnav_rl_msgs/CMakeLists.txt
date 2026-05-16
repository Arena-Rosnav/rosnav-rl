cmake_minimum_required(VERSION 3.8)
project(rosnav_rl_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Service generation ONLY
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetCommand.srv"
)

find_package(rosidl_default_runtime REQUIRED)
ament_export_dependencies(rosidl_default_runtime)
ament_package()