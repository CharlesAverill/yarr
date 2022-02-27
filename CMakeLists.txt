cmake_minimum_required(VERSION 3.13)
project(yarr LANGUAGES C CUDA)
enable_language(CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_CUDA_FLAGS "-rdc=true -g -G")

# Includes
include_directories(${CMAKE_SOURCE_DIR}/include)

# SFML stuff
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(SFML 2.5 COMPONENTS system window graphics audio network REQUIRED)
include_directories(${SFML_INCLUDE_DIR} /usr/local/include)

# Sources
file(GLOB_RECURSE SRCS ${CMAKE_SOURCE_DIR}/src/*.cu)

# Final
add_executable(yarr ${SRCS})

target_link_libraries(${PROJECT_NAME} ${SFML_LIBRARIES} csfml-graphics csfml-window curand)
