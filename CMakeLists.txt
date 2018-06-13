cmake_minimum_required(VERSION 3.5)

set (CMAKE_CXX_STANDARD 11)

find_package(WiringPi REQUIRED)
#find_package(pigpio REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${WIRINGPI_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${PIGPIO_INCLUDE_DIRS})

add_executable(LightShow main.cpp)

target_link_libraries(LightShow ${WIRINGPI_LIBRARIES})
target_link_libraries(LightShow ${PIGPIO_LIBRARIES})
target_link_libraries(LightShow ${CMAKE_THREAD_LIBS_INIT})
