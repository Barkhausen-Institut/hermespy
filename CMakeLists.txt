cmake_minimum_required(VERSION 3.15...3.26)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)


# Workaround for building shared libraries on Windows
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif()

# Configure aff3ct to only build the shared library
set(AFF3CT_COMPILE_EXE OFF CACHE INTERNAL "Disable executable target" FORCE)
set(AFF3CT_COMPILE_STATIC_LIB OFF CACHE INTERNAL "Disable static library target" FORCE)
set(AFF3CT_COMPILE_SHARED_LIB ON CACHE INTERNAL "Enable shared library target" FORCE)
set(AFF3CT_OVERRIDE_VERSION 3.0.1 CACHE INTERNAL "Enable build without .git" FORCE)
set(AFF3CT_COMPILE_OBJ ON CACHE INTERNAL "Force object building" FORCE)
set(AFF3CT_POLAR_BIT_PACKING OFF CACHE INTERNAL "Disable Polar bit packing" FORCE)

# Completely disable building the aff3ct CLI submodule
set(CLI_COMPILE_STATIC_LIB OFF CACHE INTERNAL "Disable CLI" FORCE)
set(CLI_COMPILE_SHARED_LIB OFF CACHE INTERNAL "Disable CLI" FORCE)

# Configure aff3ct submodule StreamPU
set(SPU_OVERRIDE_VERSION "0.1.0" CACHE INTERNAL "Enable SPU without .git" FORCE)
set(SPU_TESTS OFF CACHE INTERNAL "Disable SPU tests" FORCE)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(SPU_STACKTRACE OFF CACHE INTERNAL "Disable SPU" FORCE)
    set(SPU_STACKTRACE_SEGFAULT OFF CACHE INTERNAL "Disable SPU" FORCE)
endif()

# Configure aff3ct installation
add_subdirectory(submodules/affect)
set_target_properties(aff3ct-shared-lib PROPERTIES OUTPUT_NAME affect)
install(TARGETS aff3ct-shared-lib COMPONENT PythonModule)

# Configure aff3ct expansions for LDPC, Polar, RS, RSC, Turbo, and BCH
add_subdirectory(hermespy/fec/aff3ct)
set_target_properties(ldpc polar rs rsc turbo bch PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE INSTALL_RPATH "$ORIGIN")
install(TARGETS aff3ct-shared-lib ldpc polar rs rsc turbo bch DESTINATION ${SKBUILD_PROJECT_NAME}/fec/aff3ct/ COMPONENT PythonModule)

# Configure the raytracing build
# add_subdirectory(submodules/raytracing)
# set_target_properties(hermespy-rt-binding PROPERTIES OUTPUT_NAME binding)
# install(TARGETS hermespy-rt-binding LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}/channel/rt/ COMPONENT PythonModule)
