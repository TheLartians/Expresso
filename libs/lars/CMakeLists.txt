

include_directories(include)

file (GLOB headers "include/lars/*.h")

add_library(Lars EXCLUDE_FROM_ALL ${headers})
SET_TARGET_PROPERTIES(Lars PROPERTIES LINKER_LANGUAGE CXX)

set(Lars_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE STRING "Lars include dirs")


# ---- tests ----

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")

file( GLOB test_sources tests/*.cpp )

foreach( testsourcefile ${test_sources} )
  get_filename_component(filename ${testsourcefile} NAME)
  string( REPLACE ".cpp" "" testname ${filename} )
  set(exname lars_${testname}_test)
  add_executable( ${exname} ${testsourcefile} )
endforeach( testsourcefile ${test_sources} )


