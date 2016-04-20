cmake_minimum_required(VERSION 2.8)
project(VISION)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

# Where to find any custom cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)


#
# Libraries
#

# FIXME(dliu): use find_package() instead, and change references to
# <NAME>_LIBRARIES instead of <NAME>_LIB.
find_library(BOOST_FILESYSTEM_LIB boost_filesystem)
find_library(BOOST_SYSTEM_LIB boost_system)
find_library(BOOST_THREAD_LIB boost_thread)
find_library(CURL_LIB curl)
find_library(EVENT_LIB event)
find_library(FFTW3F_LIB fftw3f)
find_library(FLANN_LIB flann)
find_library(GFLAGS_LIB gflags)
find_library(GLOG_LIB glog)
find_library(GMOCK_LIB gmock)
find_library(MATIO_LIB matio)
find_library(OPENCV_CORE_LIB opencv_core)
find_library(OPENCV_FEATURES2D_LIB opencv_features2d)
find_library(OPENCV_HIGHGUI_LIB opencv_highgui)
find_library(OPENCV_IMGPROC_LIB opencv_imgproc)
find_library(OPENCV_NONFREE_LIB opencv_nonfree)
find_library(OPENCV_OBJDETECT_LIB opencv_objdetect)
find_library(OPENCV_FEATURES2D_LIB opencv_features2d)
find_library(OPENCV_NONFREE_LIB opencv_nonfree)
find_library(OPENCV_OBJDETECT_LIB opencv_objdetect)
find_library(OPENCV_CALIB3D_LIB opencv_calib3d)
#find_library(PROFILER_LIB profiler)
find_library(PROTOBUF_LIB protobuf)
find_library(PTHREAD_LIB pthread)
find_library(THRIFTNB_LIB thriftnb)
find_library(THRIFT_LIB thrift)


# FIXME(dliu): use find_package() instead.
#find_package(Thrift REQUIRED)
include_directories(/usr/include/thrift /usr/local/include/thrift)


#
# Protobuf
#

find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_BINARY_DIR}/src)  # for generated protobuf files

# NOTE(dliu): Don't change. See cmake/FindProtobuf.cmake
set(PROTOBUF_ROOT ${CMAKE_SOURCE_DIR}/src)
set(PROTOBUF_GEN_ROOT ${CMAKE_BINARY_DIR}/src)


#
# Thrift
#

# Needed for Thrift generated code to compile
# XXX: is there a better way than just hardcoding this?
add_definitions("-DHAVE_INTTYPES_H")
add_definitions("-DHAVE_NETINET_IN_H")


#
# Compiler settings
#

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

# FIXME: We need to fix the deprecation warnings (from uses of hash_set)
# For now these warnings are silenced.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
# TODO: enable all warnings

include_directories(src)
add_subdirectory(src)
