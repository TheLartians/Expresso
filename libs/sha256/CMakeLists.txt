
file (GLOB sources "*.cpp")
file (GLOB headers "include/sha256/*.h")

set (CMAKE_CXX_STANDARD 11)

add_library(sha256 ${sources} ${headers} )

set(sha256_LIBRARIES sha256 CACHE STRING "Geometry libraries")
set(sha256_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE STRING "sha256 include dirs")

