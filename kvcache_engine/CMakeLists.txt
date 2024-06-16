set(TARGET kvcache_engine)

add_library(${TARGET} STATIC
    OutputKV.cpp
    OutputKV.h
)

target_include_directories(${TARGET} PUBLIC . ${CMAKE_SOURCE_DIR})
target_compile_features(${TARGET} PUBLIC cxx_std_11)
