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
set(AFF3CT_COMPILE_EXE OFF CACHE INTERNAL "Disable executable target" FORCE)
set(AFF3CT_COMPILE_STATIC_LIB OFF CACHE INTERNAL "Disable static library target" FORCE)
set(AFF3CT_COMPILE_SHARED_LIB ON CACHE INTERNAL "Enable shared library target" FORCE)
set(AFF3CT_OVERRIDE_VERSION 3.0.1 CACHE INTERNAL "Enable build without .git" FORCE)
set(AFF3CT_COMPILE_OBJ ON CACHE INTERNAL "Force object building" FORCE)
set(AFF3ECT_INCLUDE_SPU_LIB OFF CACHE INTERNAL "Disable SPU" FORCE)
set(CLI_COMPILE_STATIC_LIB OFF CACHE INTERNAL "Disable CLI" FORCE)
set(CLI_COMPILE_SHARED_LIB ON CACHE INTERNAL "Disable CLI" FORCE)

add_subdirectory(submodules/affect)
add_subdirectory(hermespy/fec/aff3ct)

install(TARGETS aff3ct-shared-lib LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}/fec/aff3ct/ COMPONENT PythonModule)
install(TARGETS ldpc polar rs rsc turbo bch LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}/fec/aff3ct/ COMPONENT PythonModule)
