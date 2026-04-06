cmake_minimum_required( VERSION 3.19 )

set( MODULE_NAME cavallari_module )
set( MODULE_VERSION_MAJOR 0 )
set( MODULE_VERSION_MINOR 1 )
set( MODULE_VERSION "${MODULE_VERSION_MAJOR}.${MODULE_VERSION_MINOR}" )

if ( with-nest )
   set( NEST_CONFIG ${with-nest} )
else ()
   message( FATAL_ERROR "-Dwith-nest=<nest_config> is required" )
endif ()

execute_process(
    COMMAND ${NEST_CONFIG} --compiler
    RESULT_VARIABLE RES_VAR
    OUTPUT_VARIABLE NEST_COMPILER
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if ( NOT RES_VAR EQUAL 0 )
  message( FATAL_ERROR "Cannot run `${NEST_CONFIG}`. Please specify correct `nest-config` via -Dwith-nest=... " )
endif ()

set( CMAKE_CXX_COMPILER "${NEST_COMPILER}" )
project( ${MODULE_NAME} CXX )

if ( NOT CMAKE_CXX_COMPILER STREQUAL NEST_COMPILER )
  message( FATAL_ERROR
    "CMake configured the module with '${CMAKE_CXX_COMPILER}', but NEST requires '${NEST_COMPILER}'. "
    "Remove the build directory and reconfigure, or run ./install.sh so CMAKE_CXX_COMPILER is set explicitly."
  )
endif ()

execute_process(
    COMMAND ${NEST_CONFIG} --python-executable
    RESULT_VARIABLE RES_VAR
    OUTPUT_VARIABLE PYTHON_EXECUTABLE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${NEST_CONFIG} --python-version
    RESULT_VARIABLE RES_VAR
    OUTPUT_VARIABLE PYTHON_VERSION_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${NEST_CONFIG} --version
    RESULT_VARIABLE RES_VAR
    OUTPUT_VARIABLE NEST_VERSION_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${NEST_CONFIG} --prefix
    RESULT_VARIABLE RES_VAR
    OUTPUT_VARIABLE NEST_INSTALL_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${NEST_CONFIG} --cflags
    RESULT_VARIABLE RES_VAR
    OUTPUT_VARIABLE NEST_CXXFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${NEST_CONFIG} --includes
    RESULT_VARIABLE RES_VAR
    OUTPUT_VARIABLE NEST_INCLUDES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if ( NEST_INCLUDES )
  string( REPLACE " " ";" NEST_INCLUDES_LIST "${NEST_INCLUDES}" )
  foreach ( inc_complete ${NEST_INCLUDES_LIST} )
    if ( "${inc_complete}" MATCHES "^-I.*" )
      string( REGEX REPLACE "^-I(.*)" "\\1" inc "${inc_complete}" )
      if ( IS_DIRECTORY "${inc}" )
        include_directories( "${inc}" )
      endif ()
    endif ()
  endforeach ()
endif ()

execute_process(
    COMMAND ${NEST_CONFIG} --libs
    RESULT_VARIABLE RES_VAR
    OUTPUT_VARIABLE NEST_LIBS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Some conda NEST builds report a sysroot pthread path that is not present in
# the active environment. Replace it with the portable linker flag.
string( REGEX REPLACE "[^ ]*libpthread\\.so" "-lpthread" NEST_LIBS "${NEST_LIBS}" )

if ( APPLE )
    string( APPEND NEST_LIBS " -Wl,-undefined -Wl,dynamic_lookup" )
endif ()

execute_process(
    COMMAND ${NEST_CONFIG} --datadir
    RESULT_VARIABLE RES_VAR
    OUTPUT_VARIABLE NEST_INSTALL_DATADIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${NEST_CONFIG} --docdir
    RESULT_VARIABLE RES_VAR
    OUTPUT_VARIABLE NEST_INSTALL_DOCDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${NEST_CONFIG} --libdir
    RESULT_VARIABLE RES_VAR
    OUTPUT_VARIABLE NEST_LIBDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set( CMAKE_MACOSX_RPATH ON )
set( CMAKE_INSTALL_PREFIX "${NEST_INSTALL_PREFIX}" CACHE STRING "Install path prefix, prepended onto install directories." FORCE )
set( CMAKE_INSTALL_LIBDIR "${NEST_LIBDIR}" CACHE STRING "object code libraries (lib/nest or lib64/nest or lib/<multiarch-tuple>/nest on Debian)" FORCE )
set( CMAKE_INSTALL_DOCDIR "${NEST_INSTALL_DOCDIR}" CACHE STRING "documentation root (DATAROOTDIR/doc/nest)" FORCE )
set( CMAKE_INSTALL_DATADIR "${NEST_INSTALL_DATADIR}" CACHE STRING "read-only architecture-independent data (DATAROOTDIR/nest)" FORCE )

include( GNUInstallDirs )

if ( NEST_VERSION_STRING VERSION_LESS "3.9.0" )
  add_compile_definitions( CAVALLARI_USE_LEGACY_NEST_API )
endif ()

set( CPACK_GENERATOR TGZ )
set( CPACK_SOURCE_GENERATOR TGZ )

set( CPACK_PACKAGE_DESCRIPTION_SUMMARY "NEST Module ${MODULE_NAME}" )
set( CPACK_PACKAGE_VENDOR "NEST Initiative (https://www.nest-initiative.org/)" )

set( CPACK_PACKAGE_VERSION_MAJOR ${MODULE_VERSION_MAJOR} )
set( CPACK_PACKAGE_VERSION_MINOR ${MODULE_VERSION_MINOR} )
set( CPACK_PACKAGE_VERSION ${MODULE_VERSION} )

set( CPACK_SOURCE_IGNORE_FILES
    "\\\\.gitignore"
    "\\\\.git/"
    "\\\\.travis\\\\.yml"
    "/build/"
    "/_CPack_Packages/"
    "CMakeFiles/"
    "cmake_install\\\\.cmake"
    "Makefile.*"
    "CMakeCache\\\\.txt"
    "CPackConfig\\\\.cmake"
    "CPackSourceConfig\\\\.cmake"
    )
set( CPACK_SOURCE_PACKAGE_FILE_NAME ${MODULE_NAME} )

set( CPACK_PACKAGE_INSTALL_DIRECTORY "${MODULE_NAME} ${MODULE_VERSION}" )
include( CPack )

add_custom_target( dist
    COMMAND ${CMAKE_MAKE_PROGRAM} package_source
    COMMENT "Creating a source distribution from ${MODULE_NAME}..."
    )

add_subdirectory( src )

if ( ( NOT CMAKE_CROSSCOMPILING )
     AND PYTHON_EXECUTABLE
     AND ( IS_DIRECTORY ${NEST_INSTALL_PREFIX}/${NEST_INSTALL_DOCDIR}/help )
   )

    install( CODE
      "execute_process(
         COMMAND ${PYTHON_EXECUTABLE} -B generate_help.py \"${PROJECT_SOURCE_DIR}\" \"${PROJECT_BINARY_DIR}\"
         WORKING_DIRECTORY \"${NEST_INSTALL_PREFIX}/${NEST_INSTALL_DATADIR}/help_generator\"
         )"
      )

    install( DIRECTORY "${PROJECT_BINARY_DIR}/doc/help"
      DESTINATION "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DOCDIR}"
      OPTIONAL
      )

    install( CODE
      "execute_process(
         COMMAND ${PYTHON_EXECUTABLE} -B generate_helpindex.py \"${NEST_INSTALL_PREFIX}/${NEST_INSTALL_DOCDIR}\"
         WORKING_DIRECTORY \"${NEST_INSTALL_PREFIX}/${NEST_INSTALL_DATADIR}/help_generator\"
         )"
      )

  endif ()

message( "" )
message( "-------------------------------------------------------" )
message( "${MODULE_NAME} Configuration Summary" )
message( "-------------------------------------------------------" )
message( "" )
message( "C++ compiler         : ${CMAKE_CXX_COMPILER}" )
message( "NEST compiler flags  : ${NEST_CXXFLAGS}" )
message( "NEST include dirs    : ${NEST_INCLUDES}" )
message( "NEST libraries flags : ${NEST_LIBS}" )
message( "" )
message( "-------------------------------------------------------" )
message( "" )
message( "You can now build and install '${MODULE_NAME}' using" )
message( "  make" )
message( "  make install" )
message( "" )
if ( "${CMAKE_HOST_SYSTEM_NAME}" MATCHES "^Darwin.*" )
  message( "The library file lib${MODULE_NAME}.dylib will be installed to" )
else ()
  message( "The library file lib${MODULE_NAME}.so will be installed to" )
endif()
message( "  ${CMAKE_INSTALL_FULL_LIBDIR}/nest/" )
message( "Help files will be installed to" )
message( "  ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DOCDIR}" )
message( "" )
message( "The module can be loaded into NEST using" )
message( "  nest.Install('${MODULE_NAME}')  (in PyNEST)" )
message( "" )

if( NOT "${CMAKE_INSTALL_PREFIX}" STREQUAL "${NEST_INSTALL_PREFIX}" )
  message( "The module will be installed into a non-default location!" )
  message( "Make sure to set the environment variables:" )
  if ( "${CMAKE_HOST_SYSTEM_NAME}" MATCHES "^Darwin.*" )
    message( "  export DYLD_LIBRARY_PATH=${CMAKE_INSTALL_FULL_LIBDIR}/nest:$DYLD_LIBRARY_PATH" )
  else ()
    message( "  export LD_LIBRARY_PATH=${CMAKE_INSTALL_FULL_LIBDIR}/nest:$LD_LIBRARY_PATH" )
  endif()
  message( "" )
endif ()
