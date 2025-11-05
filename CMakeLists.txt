cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 14)
project(shape_based_matching)


# debug or release
SET(CMAKE_BUILD_TYPE "Release")


# arm or x86
IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    IF(${CMAKE_SIZEOF_VOID_P} EQUAL 4)
        # 32-bit ARM
        SET(PLATFORM_COMPILE_FLAGS "-mfpu=neon")
    ELSE()
        # 64-bit ARM (ARM64)
        SET(PLATFORM_COMPILE_FLAGS "-march=native")
    ENDIF()
ELSE()
    SET(PLATFORM_COMPILE_FLAGS "-march=native")

    # some places of the algorithm are designed for 128 SIMD
    # so 128 SSE may slightly faster than 256 AVX, you may want this
#    SET(PLATFORM_COMPILE_FLAGS "-msse -msse2 -msse3 -msse4 -mssse3")  # SSE only
ENDIF()

# SET(PLATFORM_COMPILE_FLAGS "-DMIPP_NO_INTRINSICS")  # close SIMD
SET(COMMON_COMPILE_FLAGS "-Wall -Wno-sign-compare")
SET(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS} $ENV{CXXFLAGS}")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")


# opencv
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# include MIPP headers
include_directories (${INCLUDE_DIRECTORIES} "${CMAKE_CURRENT_SOURCE_DIR}/MIPP/")


# shared library
add_library(line2dup SHARED line2Dup.cpp)
target_link_libraries(line2dup ${OpenCV_LIBS})

# test exe
add_executable(${PROJECT_NAME}_test line2Dup.cpp test.cpp)
target_link_libraries(${PROJECT_NAME}_test ${OpenCV_LIBS})

