cmake_minimum_required(VERSION 3.5)

#find_package(WiringPi REQUIRED)
# find_package(pigpio REQUIRED)
find_package(Threads REQUIRED)
find_package( OpenCV REQUIRED )
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

# set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-pg")

#include_directories(${WIRINGPI_INCLUDE_DIRS})
include_directories(${pigpio_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
include_directories("/home/pi/tensorflow")
include_directories("/home/pi/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include/")
include_directories("/opt/vc/src/hello_pi/hello_fft")
file(GLOB_RECURSE fft_src "/opt/vc/src/hello_pi/hello_fft/*.c")
list(REMOVE_ITEM fft_src "/opt/vc/src/hello_pi/hello_fft/hello_fft.c")
list(REMOVE_ITEM fft_src "/opt/vc/src/hello_pi/hello_fft/hello_fft_2d.c")

# SET_SOURCE_FILES_PROPERTIES(${fft_src} PROPERTIES LANGUAGE CXX)

# add_library(gpu_fft ${fft_src})

add_executable(Test main.cpp ${fft_src})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#target_link_libraries(LightShow ${WIRINGPI_LIBRARIES})
# target_link_libraries(Test ${pigpio_LIBRARY})
target_link_libraries(Test ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS})
target_link_libraries(Test rt dl)
target_link_libraries(Test Aquila)
target_link_libraries(Test fftw3)
target_link_libraries(Test ${CMAKE_SOURCE_DIR}/libtensorflow-lite.a)