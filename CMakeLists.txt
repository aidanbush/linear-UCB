cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(run_test)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -g -Wall")

add_executable(run_test run_test.cpp linUCB.cpp environment.cpp)

target_link_libraries(run_test "${TORCH_LIBRARIES}")
set_property(TARGET run_test PROPERTY CXX_STANDARD 20)
