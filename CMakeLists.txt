cmake_minimum_required(VERSION 3.5)

#find_package(WiringPi REQUIRED)
find_package(pigpio REQUIRED)
find_package(Threads REQUIRED)

#include_directories(${WIRINGPI_INCLUDE_DIRS})
include_directories(${pigpio_INCLUDE_DIRS})

add_executable(LightShow main.cpp)

#target_link_libraries(LightShow ${WIRINGPI_LIBRARIES})
target_link_libraries(LightShow ${pigpio_LIBRARY})
target_link_libraries(LightShow ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(LightShow rt)
