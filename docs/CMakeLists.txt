#---- Find Doxygen.
string(JSON Doxygen_REQUIRED_VERSION GET "${VERSION_JSON}" dependencies doxygen value)

find_package(
    Doxygen
    ${Doxygen_REQUIRED_VERSION}
    REQUIRED
)

#---- Our 'homepage' is the README file.
set(HOMEPAGE_MD ${CMAKE_SOURCE_DIR}/README.md)

#---- Doxygen settings.
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${HOMEPAGE_MD})
set(DOXYGEN_TIMESTAMP YES)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_WARN_AS_ERROR YES)
set(DOXYGEN_SOURCE_BROWSER YES)
set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)

#---- Extract files for documentation.
get_property(GraphDispatching_FILES_FOR_DOC GLOBAL PROPERTY GraphDispatching_FILES_FOR_DOC)
list(LENGTH GraphDispatching_FILES_FOR_DOC GraphDispatching_FILES_FOR_DOC_LENGTH)
if(NOT GraphDispatching_FILES_FOR_DOC_LENGTH EQUAL 31)
    message(FATAL_ERROR "You have ${GraphDispatching_FILES_FOR_DOC_LENGTH} files for documentation.")
endif()

#---- Add Doxygen as a target.
#     See also https://cmake.org/cmake/help/latest/module/FindDoxygen.html.
doxygen_add_docs(
    docs

    # Homepage
    ${HOMEPAGE_MD}

    # Files added to the global property KokkosUtils_FILES_FOR_DOC
    ${GraphDispatching_FILES_FOR_DOC}

    # Files NOT added to the global property KokkosUtils_FILES_FOR_DOC
    ${CMAKE_SOURCE_DIR}/docs/tests.dox
)
