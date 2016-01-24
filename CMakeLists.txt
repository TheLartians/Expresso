
cmake_minimum_required (VERSION 2.6)

if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
      message(FATAL_ERROR "Source and build directories cannot be the same.")
endif()

project (Symbols)

# ---- add sources ----

set (PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source)
set (PROJECT_LIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs)
set (PROJECT_PYSOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pysource)

file (GLOB sources ${PROJECT_SOURCE_DIR}/expression/*.cpp)
file (GLOB headers ${PROJECT_SOURCE_DIR}/expression/*.h)

# ---- enable c++11 ----
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# ---- python ----
SET(Python_ADDITIONAL_VERSIONS 2.7 2.6 2.5)
FIND_PACKAGE(PythonInterp REQUIRED)
FIND_PACKAGE(PythonLibs REQUIRED)
FIND_PACKAGE(Boost COMPONENTS python REQUIRED)

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})
LINK_LIBRARIES(${Boost_LIBRARIES} ${PYTHON_LIBRARIES}) # Deprecated but so convenient!

# ---- create module ----

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC"  )
add_library(symbols STATIC ${sources} ${headers} )
PYTHON_ADD_MODULE(_symbols ${PROJECT_SOURCE_DIR}/python.cpp)
TARGET_LINK_LIBRARIES(_symbols symbols)

include_directories (${PROJECT_LIBS_DIR})

# ---- libs -----

add_subdirectory (${PROJECT_LIBS_DIR}/sha256)
target_link_libraries(symbols sha256)

# ---- lars libs ----

add_subdirectory (${PROJECT_LIBS_DIR}/lars)

# ---- install ----

install(TARGETS _symbols DESTINATION pysymbols)
install(DIRECTORY ${PROJECT_PYSOURCE_DIR}/ DESTINATION .)

