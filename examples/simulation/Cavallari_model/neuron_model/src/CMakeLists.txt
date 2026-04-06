set(
  MODULE_SOURCES
  cavallari_module.cpp
  iaf_bw_2003.h
  iaf_bw_2003.cpp
)

add_library(${MODULE_NAME}_module MODULE ${MODULE_SOURCES})
set_target_properties(
  ${MODULE_NAME}_module
  PROPERTIES
  COMPILE_FLAGS "${NEST_CXXFLAGS} -DLTX_MODULE"
  LINK_FLAGS "${NEST_LIBS}"
  PREFIX ""
  OUTPUT_NAME ${MODULE_NAME}
)

install(
  TARGETS ${MODULE_NAME}_module
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/nest
)
