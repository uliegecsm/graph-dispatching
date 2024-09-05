#---- Find HPX.
string(JSON HPX_REQUIRED_VERSION  GET "${VERSION_JSON}" dependencies hpx value)

find_package(
    HPX
    ${HPX_REQUIRED_VERSION}
    CONFIG
    REQUIRED
)

# Add helper files to the documentation.
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/Helpers.hpp)

# Add subdiretories.
add_subdirectory(adaptors)

add_one_test(NAME HelloWorld ADDITIONAL_LINK_LIBRARIES HPX::hpx)
