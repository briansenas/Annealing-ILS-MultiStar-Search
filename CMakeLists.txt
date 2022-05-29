PROJECT(CXX)
cmake_minimum_required(VERSION 3.16)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_executable(Recocido src/Recocido.cpp src/Util_Euclidean.cpp
    src/Util_Genetics.cpp src/Util_mytools.cpp src/Util_ReadData.cpp)
add_executable(BMB src/BMB.cpp src/Util_Euclidean.cpp
    src/Util_Genetics.cpp src/Util_mytools.cpp src/Util_ReadData.cpp)
add_executable(ILS src/ILS.cpp src/Util_Euclidean.cpp
    src/Util_Genetics.cpp src/Util_mytools.cpp src/Util_ReadData.cpp)
