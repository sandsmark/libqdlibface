cmake_minimum_required(VERSION 3.2.0)
project(libqdlibface LANGUAGES CXX VERSION 1.0)

find_package(dlib REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Gui)

set(sources
    Face.cpp
    Database.cpp
    )

add_library(libqdlibface ${sources})
target_link_libraries(libqdlibface Qt5::Gui Qt5::Core dlib::dlib)

