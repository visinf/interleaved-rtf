include_directories (${PROJECT_SOURCE_DIR})

file(GLOB app_srcs *.cpp)
file(GLOB app_hdrs *.h)

add_executable (Deblur ${app_srcs} ${app_hdrs})
