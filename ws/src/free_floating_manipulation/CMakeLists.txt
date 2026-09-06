cmake_minimum_required(VERSION 3.22)
project(free_floating_manipulation)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# Every dependency is found REQUIRED, so a missing one fails the build here
# rather than at the first line of code that needs it.
find_package(behaviortree_cpp REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(urdf REQUIRED)
find_package(srdfdom REQUIRED)
find_package(Eigen3 REQUIRED)

message(STATUS "BehaviorTree.CPP ${behaviortree_cpp_VERSION}")
message(STATUS "MoveIt core      ${moveit_core_VERSION}")

add_library(free_floating_dynamics SHARED src/dynamics.cpp src/trajectory_validity.cpp)
target_include_directories(free_floating_dynamics PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(free_floating_dynamics PUBLIC Eigen3::Eigen)
ament_target_dependencies(free_floating_dynamics PUBLIC moveit_core urdf srdfdom)
target_link_libraries(free_floating_dynamics PUBLIC ${moveit_core_LIBRARIES})

# Writes the dynamics over a set of configurations so the Python
# implementation can be compared against it block by block.
add_executable(dump_dynamics src/dump_dynamics.cpp)
target_link_libraries(dump_dynamics free_floating_dynamics)
ament_target_dependencies(dump_dynamics moveit_core urdf srdfdom)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS free_floating_dynamics
  EXPORT export_free_floating_manipulation
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)
add_executable(servicing_demo src/servicing_demo.cpp)
target_link_libraries(servicing_demo free_floating_dynamics)
# OMPL is pulled in for one call, RNG::setSeed, so the demonstration is
# repeatable. MoveIt exposes no seed of its own.
find_package(ompl REQUIRED)
target_include_directories(servicing_demo PRIVATE ${OMPL_INCLUDE_DIRS})
target_link_libraries(servicing_demo ${OMPL_LIBRARIES})
ament_target_dependencies(servicing_demo moveit_core moveit_ros_planning rclcpp
  pluginlib urdf srdfdom geometry_msgs moveit_msgs)

install(TARGETS dump_dynamics servicing_demo DESTINATION lib/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(free_floating_dynamics)
ament_export_dependencies(moveit_core urdf srdfdom Eigen3)

ament_package()
