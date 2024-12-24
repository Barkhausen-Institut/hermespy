cmake_minimum_required(VERSION 3.15...3.19)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)


# Workaround for building shared libraries on Windows
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif()

# Configure aff3ct only build the shared library
# This builds and tests on unix, unclear on windows
set(AFF3CT_COMPILE_EXE OFF CACHE INTERNAL "Disable executable target" FORCE)
set(AFF3CT_COMPILE_STATIC_LIB OFF CACHE INTERNAL "Disable static library target" FORCE)
set(AFF3CT_COMPILE_SHARED_LIB ON CACHE INTERNAL "Enable shared library target" FORCE)
set(AFF3CT_OVERRIDE_VERSION 3.0.1 CACHE INTERNAL "Enable build without .git" FORCE)
set(AFF3CT_COMPILE_OBJ OFF CACHE INTERNAL "Force object building" FORCE)
set(CLI_COMPILE_STATIC_LIB OFF CACHE INTERNAL "Disable CLI" FORCE)
set(CLI_COMPILE_SHARED_LIB OFF CACHE INTERNAL "Disable CLI" FORCE)
set(SPU_OVERRIDE_VERSION "0.1.0" CACHE INTERNAL FROCE)

add_subdirectory(submodules/affect)
add_subdirectory(hermespy/fec/aff3ct)

# This is required in order to find the aff3ct shared lib in editable unix installs
install(TARGETS aff3ct-shared-lib LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}/fec/aff3ct/ COMPONENT PythonModule)
set_target_properties(aff3ct-shared-lib PROPERTIES OUTPUT_NAME "affect")
set_target_properties(ldpc polar rs rsc turbo bch PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE INSTALL_RPATH "$ORIGIN")


install(TARGETS ldpc polar rs rsc turbo bch LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}/fec/aff3ct/ COMPONENT PythonModule)
