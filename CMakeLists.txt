# Require CMake 3.15+ (matching scikit-build-core)
# Use new versions of all policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})

set(PROJECT_LANGUAGES CXX)  # Start with just CXX as the default language
set(CMAKE_CXX_STANDARD 17)  # Set the C++ standard to C++17

if (WIN32)
    set (CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif ()

# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
find_package(Python REQUIRED COMPONENTS Interpreter)

file(GLOB_RECURSE SOURCES shppy/_core/*.cpp)
list(REMOVE_ITEM SOURCES shppy/_core/main.cpp)

add_library(_C SHARED ${SOURCES})

set_target_properties(_C PROPERTIES POSITION_INDEPENDENT_CODE ON LIBRARY_OUTPUT_NAME "shppy" OUTPUT_NAME "libshppy")
target_compile_definitions(_C PRIVATE VERSION_INFO=${PROJECT_VERSION})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_executable(_debug shppy/_core/main.cpp)
    target_link_libraries(_debug PRIVATE _C)
    target_compile_definitions(_debug PRIVATE DEBUG)
endif()

# Install the library
if(NOT "${SKBUILD_PROJECT_NAME}" STREQUAL "")
    # Install _C to site-packages
    install(TARGETS _C LIBRARY DESTINATION shppy RUNTIME DESTINATION shppy)
    # Install tools scripts to bin with shppy- prefix to bin
    file(GLOB TOOL_SCRIPTS "tools/*.py")
    foreach(SCRIPT ${TOOL_SCRIPTS})
        get_filename_component(SCRIPT_NAME ${SCRIPT} NAME_WE)
        install(PROGRAMS ${SCRIPT} DESTINATION ../scripts RENAME "shppy-${SCRIPT_NAME}")
    endforeach()
else()
    add_custom_command(TARGET _C POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_C> ${CMAKE_CURRENT_SOURCE_DIR}/shppy/
    )
endif()

# file(GLOB TOOL_SCRIPTS "tools/*.py")
# foreach(SCRIPT ${TOOL_SCRIPTS})
#     get_filename_component(SCRIPT_NAME ${SCRIPT} NAME_WE)
#     install(FILES ${SCRIPT}
#             DESTINATION  ${CMAKE_INSTALL_PREFIX}/bin
#             RENAME "shppy-${SCRIPT_NAME}"
#             PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE 
#                        GROUP_READ GROUP_EXECUTE
#                        WORLD_READ WORLD_EXECUTE)
# endforeach()


# ====== ADDITION FOR EXECUTABLE GEm NERATION ======

# Create an executable from main.cpp
# file(GLOB_RECURSE MAIN_SOURCES "f 