cmake_minimum_required(VERSION 3.14)
project(yarr LANGUAGES C CUDA CXX)
enable_language(CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_CUDA_FLAGS "-rdc=true")

# Absolute paths
add_compile_definitions(YARR_ROOT="${CMAKE_SOURCE_DIR}/")
add_compile_definitions(ASSETS_ROOT="${CMAKE_SOURCE_DIR}/assets/")

# Debug flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

# Headers
include_directories("${CMAKE_SOURCE_DIR}/include")

# Resource Compilation
include(cmake/CMakeRC.cmake)
file(GLOB_RECURSE ASSETS "${CMAKE_SOURCE_DIR}/assets")
cmrc_add_resource_library(yarr-assets ALIAS yarr::rc NAMESPACE yarr ${ASSETS})

# SFML
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(SFML 2.5 COMPONENTS system window graphics audio network REQUIRED)
include_directories(${SFML_INCLUDE_DIR} "/usr/local/include")

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Sources
file(GLOB_RECURSE SRCS ${CMAKE_SOURCE_DIR}/src/*.cu)

# Final
add_executable(yarr ${SRCS})

target_link_libraries(
    ${PROJECT_NAME}
    ${SFML_LIBRARIES} csfml-graphics csfml-window
    curand
    yarr::rc
    ${OpenCV_LIBS}
)
