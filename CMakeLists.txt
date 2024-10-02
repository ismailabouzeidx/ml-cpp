cmake_minimum_required(VERSION 3.10)
project(ml_cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(OpenCV REQUIRED)

file(GLOB SOURCES 
    "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_executable(ml_cpp ${SOURCES})

# Link the OpenCV libraries
target_link_libraries(ml_cpp ${OpenCV_LIBS})