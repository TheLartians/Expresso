
file (GLOB sources "*.cpp")
file (GLOB headers "*.h")

set (CMAKE_CXX_STANDARD 11)

add_library(sha256 ${sources} ${headers} )

