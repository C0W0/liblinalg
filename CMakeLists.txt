cmake_minimum_required(VERSION 3.24)
project(linalg)

set(CMAKE_CXX_STANDARD 17)

add_library(linalg STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Matrix.cpp)

target_include_directories(linalg PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(DIST_DIR "${CMAKE_CURRENT_BINARY_DIR}/dist")

add_custom_command(TARGET linalg POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/include ${DIST_DIR}/include
)

# Custom command to copy the compiled binary
add_custom_command(TARGET linalg POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:linalg> ${DIST_DIR}/lib/$<TARGET_FILE_NAME:linalg>
)

add_executable(linalg_test main.cpp
        src/Matrix.cpp
        include/Matrix.h)

target_include_directories(linalg_test PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)