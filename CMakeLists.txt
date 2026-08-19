cmake_minimum_required(VERSION 3.16)
project(physicar_autonomy)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(physicar_interfaces REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(lane_follow_node src/lane_follow_node.cpp)
ament_target_dependencies(lane_follow_node
  cv_bridge
  physicar_interfaces
  rcl_interfaces
  rclcpp
  sensor_msgs
  std_msgs
)
target_link_libraries(lane_follow_node ${OpenCV_LIBS})

add_executable(lidar_obstacle_node src/lidar_obstacle_node.cpp)
ament_target_dependencies(lidar_obstacle_node
  geometry_msgs
  physicar_interfaces
  rcl_interfaces
  rclcpp
  sensor_msgs
  visualization_msgs
)

install(TARGETS lane_follow_node lidar_obstacle_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
