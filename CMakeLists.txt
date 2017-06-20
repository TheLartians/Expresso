
cmake_minimum_required (VERSION 2.6)

if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
      message(FATAL_ERROR "Source and build directories cannot be the same.")
endif()

project(Expresso)

# ---- add sources ----

set (PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source)
set (PROJECT_INCUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set (PROJECT_LIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs)
set (PROJECT_PYSOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/expresso)

file (GLOB sources ${PROJECT_SOURCE_DIR}/expresso/*.cpp)
file (GLOB headers ${PROJECT_INCUDE_DIR}/expresso/*.h)
file (GLOB_RECURSE pysources ${PROJECT_PYSOURCES_DIR}/*.py)

# ---- group python sources ----

foreach (resource ${pysources})
get_filename_component(currentdir ${resource} PATH)
list (APPEND includeDirs ${currentdir})
get_filename_component(relative ${currentdir} PATH)
STRING(REGEX REPLACE ${CMAKE_SOURCE_DIR} "" groupname "${currentdir}")
STRING(REGEX REPLACE "/" "\\\\" groupname "${groupname}")
source_group(${groupname} FILES ${resource})
endforeach()

# ---- enable c++11 ----
if(NOT CMAKE_CXX_STANDARD GREATER 11)
  set (CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

# ---- create module ----

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC"  )
add_library(expresso STATIC ${sources} ${headers} )

# ---- python ----
if(CREATE_PYTHON_TARGET)

  SET(Python_ADDITIONAL_VERSIONS 2.7 2.6 2.5)
  FIND_PACKAGE(PythonInterp REQUIRED)
  FIND_PACKAGE(PythonLibs REQUIRED)
  FIND_PACKAGE(Boost COMPONENTS python REQUIRED)

  PYTHON_ADD_MODULE(_expresso ${PROJECT_SOURCE_DIR}/python.cpp ${pysources})
  
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(_expresso ${Boost_LIBRARIES} ${PYTHON_LIBRARIES}) # Deprecated but so convenient!
  TARGET_LINK_LIBRARIES(_expresso expresso)

  install(TARGETS _expresso DESTINATION expresso)
  install(DIRECTORY ${PROJECT_PYSOURCES_DIR} DESTINATION .)

endif(CREATE_PYTHON_TARGET)

# ---- libs -----

include_directories (${PROJECT_INCUDE_DIR})
include_directories (${PROJECT_LIBS_DIR})

if(NOT sha256_LIBRARIES)
  add_subdirectory (${PROJECT_LIBS_DIR}/sha256)
endif(NOT sha256_LIBRARIES)

 include_directories(${sha256_INCLUDE_DIRS})
target_link_libraries(expresso ${sha256_LIBRARIES})


# ---- lars libs ----

IF(LARS_LIBRARY)
  SET(lars_include_dirs ${lars_include_dirs} "${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)
  SET(lars_libraries ${lars_libraries} ${sha256_LIBRARIES} expresso PARENT_SCOPE)
  SET(Lars_INCLUDE_DIRS ${lars_include_dirs})
ELSEIF()
  add_subdirectory (${PROJECT_LIBS_DIR}/lars)
ENDIF()

include_directories(${Lars_INCLUDE_DIRS})


# ---- export ----

set(Expresso_LIBRARIES expresso CACHE STRING "Expresso libraries")
set(Expresso_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include ${Lars_INCLUDE_DIRS} ${sha256_INCLUDE_DIRS} CACHE STRING "Expresso include dirs")


install (TARGETS expresso DESTINATION lib)


