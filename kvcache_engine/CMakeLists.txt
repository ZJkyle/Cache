add_library(kvcache_engine
            MyClass.cpp
            MyClass.h)

target_include_directories(kvcache_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
