cmake_minimum_required(VERSION 3.22.1)

project(storm_pick_place_demo LANGUAGES CXX)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/pick_place_task_manager_node
  scripts/object_manager_node
  DESTINATION lib/${PROJECT_NAME}
)

include(CTest)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_auto_find_test_dependencies()
endif()

ament_auto_package(INSTALL_TO_SHARE config launch rviz urdf)

