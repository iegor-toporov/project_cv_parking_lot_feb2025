cmake_minimum_required(VERSION 3.10)

project(MyProject)

find_package(OpenCV REQUIRED)

add_executable(MyProject main.cpp help.cpp pugixml-1.15/src/pugixml.cpp)

add_library(pugixml STATIC build/pugixml-1.15/src/pugixml.cpp)

target_include_directories(MyProject PRIVATE ${OpenCV_INCLUDE_DIRS} build/pugixml-1.15/src)

target_link_libraries(MyProject PRIVATE ${OpenCV_LIBS})

include_directories("pugixml")
