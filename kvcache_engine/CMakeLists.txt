set(TARGET kvcache_engine)

add_library(${TARGET} STATIC
    MyClass.cpp
    MyClass.h
)
# if (BUILD_SHARED_LIBS)
#     set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
# endif()

target_include_directories(${TARGET} PUBLIC .)
target_compile_features(${TARGET} PUBLIC cxx_std_11)
