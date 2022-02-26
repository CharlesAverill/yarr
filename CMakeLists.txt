cmake_minimum_required(VERSION 3.13)

project(yarr LANGUAGES C CUDA)
enable_language(CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_CUDA_FLAGS "-rdc=true -g -G")

# Find OpenGL dependencies
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(yarr src/main.cu src/utils.cu src/canvas.cu src/renderobjects/triangle.cu src/renderobjects/renderobject.cu)
target_link_libraries(${PROJECT_NAME} OpenGL::GL GLUT::GLUT)
