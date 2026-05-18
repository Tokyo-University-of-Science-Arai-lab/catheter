cmake_minimum_required(VERSION 3.8)
project(xarm7_moveit_cpp)

# =============================
# Compiler options
# =============================
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# =============================
# Dependencies
# =============================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(controller_manager_msgs REQUIRED)
find_package(xarm7_moveit_msgs REQUIRED)



# =============================
# Executables
# =============================

# ---- legacy single-shot node（残すなら）
add_executable(move_tcp_x
  src/move_tcp_x.cpp
)

ament_target_dependencies(
  move_tcp_x
  rclcpp
  moveit_ros_planning_interface
  geometry_msgs
  sensor_msgs
  controller_manager_msgs
)



# ---- Action Server（本命）
add_executable(move_tcp_action_server
  src/move_tcp_action_server.cpp
)

ament_target_dependencies(
  move_tcp_action_server
  rclcpp
  rclcpp_action
  moveit_ros_planning_interface
  geometry_msgs
  sensor_msgs
  controller_manager_msgs
  xarm7_moveit_msgs 
)



# =============================
# Install
# =============================
install(
  TARGETS
    move_tcp_x
    move_tcp_action_server
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# =============================
# Package
# =============================
ament_package()
