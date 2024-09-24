cmake_minimum_required(VERSION 3.20)
project(dnn LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "-std=c++17")

# set Boost
set(BOOST_ROOT /home/systembug/workspace/boost/boost_1_86_0)
set(BOOST_INCLUDEDIR ${BOOST_ROOT})
set(BOOST_LIBRARYDIR ${BOOST_ROOT}/stage/lib)

find_package(Boost COMPONENTS system filesystem log log_setup thread REQUIRED)

if (Boost_FOUND)
    message(STATUS "Boost found successfully!")
    message(STATUS "Boost's include directory: " ${Boost_INCLUDE_DIRS})
    include_directories(${Boost_INCLUDE_DIRS})
    message(STATUS "Boost's library directory: " ${Boost_LIBRARY_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
endif()

set(CMAKE_CXX_FLAGS "-DBOOST_LOG_DYN_LINK")

# set OpenCV
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(OpenCV_DIR /home/systembug/workspace/opencv/opencv-4.10.0/cmake-build-debug)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(OpenCV_DIR /home/systembug/workspace/opencv/opencv-4.10.0/cmake-build-release)
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# set cudnn
set(CUDNN_INCLUDE_DIR "/usr/include")
set(CUDNN_LIBRARY_DIR "/usr/lib/x86_64-linux-gnu")
set(CUDNN_LIBRARY ${CUDNN_LIBRARY_DIR}/libcudnn.so)
include_directories(${CUDNN_INCLUDE_DIR})
link_directories(${CUDNN_LIBRARY_DIR})

# set libtorch
find_package(Torch QUIET PATHS "/home/systembug/workspace/libtorch")
include_directories(${TORCH_INCLUDE_DIRS})

# set yaml-cpp
include(FetchContent)
FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG 0.8.0 # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
)
FetchContent_MakeAvailable(yaml-cpp)

add_executable(dnn main.cpp InferLibtorch.cpp InferONNX.cpp InferNothing.cpp InferManager.cpp ConfigManager.cpp ConfigManager.h)
target_link_libraries(dnn PUBLIC yaml-cpp::yaml-cpp ${Boost_LIBRARIES} ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${CUDNN_LIBRARY} Boost::log Boost::system Boost::filesystem Boost::thread)

set(BOOST_LIBRARY_FILES
        ${BOOST_LIBRARYDIR}/libboost_filesystem.so.1.86.0
        ${BOOST_LIBRARYDIR}/libboost_thread.so.1.86.0
)

foreach(file ${BOOST_LIBRARY_FILES})
    add_custom_command(
            TARGET dnn
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${file}
            ${CMAKE_BINARY_DIR}
            COMMENT "Copying ${file} to run binary after build"
    )
endforeach()

message(STATUS "cuDNN libraries: ${CUDNN_LIBRARY_DIR}/libcudnn.so")