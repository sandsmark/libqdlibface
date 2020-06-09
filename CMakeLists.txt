cmake_minimum_required(VERSION 3.2.0)
project(libqdlibface LANGUAGES CXX VERSION 1.0)

set(QDLIBFACE_MAJOR_VERSION 0)
set(QDLIBFACE_MINOR_VERSION 1)
set(QDLIBFACE_PATCH_VERSION 0)
set(QDLIBFACE_VERSION
  ${QDLIBFACE_MAJOR_VERSION}.${QDLIBFACE_MINOR_VERSION}.${QDLIBFACE_PATCH_VERSION})

find_package(dlib REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Gui)
find_package(Threads REQUIRED)

set(sources
    Face.cpp
    Database.cpp
    )

configure_file(QDLibFaceConfig.cmake.in
  "${PROJECT_BINARY_DIR}/QDLibFaceConfig.cmake" @ONLY)

add_library(libqdlibface ${sources})
target_link_libraries(libqdlibface PUBLIC Qt5::Gui Qt5::Core dlib::dlib)

include(GNUInstallDirs)
set_target_properties(libqdlibface PROPERTIES
    PUBLIC_HEADER "Face.h;Database.h"
    )

install(TARGETS libqdlibface
  EXPORT qdlibface
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/qdlibface"
    COMPONENT dev)

set(ConfigPackageLocation "${CMAKE_INSTALL_LIBDIR}/cmake/qdlibface")
install(EXPORT qdlibface
    NAMESPACE qdlibface::
    DESTINATION ${ConfigPackageLocation})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
         "${CMAKE_CURRENT_BINARY_DIR}/QDLibFaceConfigVersion.cmake"
         VERSION ${QDLIBFACE_VERSION}
         COMPATIBILITY AnyNewerVersion
         )

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/QDLibFaceConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/QDLibFaceConfigVersion.cmake"
    DESTINATION ${ConfigPackageLocation})

