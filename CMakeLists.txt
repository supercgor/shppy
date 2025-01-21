# Require CMake 3.15+ (matching scikit-build-core)
# Use new versions of all policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})

set(PROJECT_LANGUAGES CXX)  # Start with just CXX as the default language
set(CMAKE_CXX_STANDARD 17)  # Set the C++ standard to C++17

# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
find_package(Python REQUIRED COMPONENTS Interpreter)

file(GLOB_RECURSE SOURCES shppy/_core/*.cpp)
list(REMOVE_ITEM SOURCES shppy/_core/main.cpp)

add_library(_C SHARED ${SOURCES})

set_target_properties(_C PROPERTIES POSITION_INDEPENDENT_CODE ON LIBRARY_OUTPUT_NAME "shppy" OUTPUT_NAME "shppy")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_executable(_debug shppy/_core/main.cpp)
    target_link_libraries(_debug PRIVATE _C)
    target_compile_definitions(_debug PRIVATE DEBUG)
endif()

add_custom_command(TARGET _C POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_C> ${CMAKE_CURRENT_SOURCE_DIR}/shppy/
)

target_compile_definitions(_C PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _C DESTINATION ${CMAKE_INSTALL_PREFIX})

# ====== ADDITION FOR EXECUTABLE GENERATION ======

# Create an executable from main.cpp
# file(GLOB_RECURSE MAIN_SOURCES "fisea/_C/main.cpp")

# if(CUDAToolkit_FOUND)
#   add_executable(fisea_exe ${MAIN_SOURCES} ${CUDA_SOURCES})
#   target_link_libraries(fisea_exe PRIVATE CUDA::cudart _CU)
# else()
#   add_executable(fisea_exe ${MAIN_SOURCES})
# endif()

# # Link pybind11 and other necessary libraries if required
# target_link_libraries(fisea_exe PRIVATE pybind11::headers)

# # Define properties for the executable
# set_target_properties(fisea_exe PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
#     OUTPUT_NAME "fisea_exe"
# )