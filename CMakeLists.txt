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
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

add_library(local_candidate_planner src/local_candidate_planner.cpp)
target_include_directories(local_candidate_planner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(lane_follow_node src/lane_follow_node.cpp)
target_include_directories(lane_follow_node PRIVATE include)
ament_target_dependencies(lane_follow_node
  cv_bridge
  geometry_msgs
  physicar_interfaces
  rcl_interfaces
  rclcpp
  sensor_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
)
target_link_libraries(lane_follow_node local_candidate_planner ${OpenCV_LIBS})

add_executable(lidar_obstacle_node src/lidar_obstacle_node.cpp)
ament_target_dependencies(lidar_obstacle_node
  geometry_msgs
  physicar_interfaces
  rcl_interfaces
  rclcpp
  sensor_msgs
  visualization_msgs
)

install(TARGETS lane_follow_node lidar_obstacle_node local_candidate_planner
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_local_candidate_planner
    test/test_local_candidate_planner.cpp
  )
  target_include_directories(test_local_candidate_planner PRIVATE include)
  target_link_libraries(test_local_candidate_planner local_candidate_planner)
endif()

ament_package()
