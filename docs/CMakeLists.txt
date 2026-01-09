#---- Find Doxygen.
string(JSON Doxygen_REQUIRED_VERSION GET "${VERSION_JSON}" dependencies doxygen value)

find_package(
    Doxygen
    ${Doxygen_REQUIRED_VERSION}
    REQUIRED
)

#---- Fetch tag files of other projects.
set(KokkosUtils_DOC_URL https://uliegecsm.github.io/kokkos-utils/)
file(DOWNLOAD ${KokkosUtils_DOC_URL}/kokkos-utils.tag ${CMAKE_CURRENT_BINARY_DIR}/kokkos-utils.tag STATUS fetch_tag_file)
if(NOT fetch_tag_file EQUAL 0)
    message(FATAL_ERROR "An error occurred while fetching tag file for 'kokkos-utils' from ${KokkosUtils_DOC_URL} (${fetch_tag_file})")
endif()
set(DOXYGEN_TAGFILES ${CMAKE_CURRENT_BINARY_DIR}/kokkos-utils.tag=${KokkosUtils_DOC_URL})

#---- Our 'homepage' is the README file.
set(HOMEPAGE_MD ${CMAKE_SOURCE_DIR}/README.md)

#---- Doxygen settings.
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_CITE_BIB_FILES ${CMAKE_CURRENT_SOURCE_DIR}/references.bib)
set(DOXYGEN_ENABLE_PREPROCESSING YES)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_EXTRACT_LOCAL_CLASSES YES)
set(DOXYGEN_EXTRACT_PACKAGE YES)
set(DOXYGEN_EXTRACT_PRIVATE YES)
set(DOXYGEN_EXTRACT_STATIC YES)
set(DOXYGEN_GENERATE_TESTLIST YES)
set(DOXYGEN_MACRO_EXPANSION YES)
set(DOXYGEN_REFERENCES_RELATION YES)
set(DOXYGEN_SOURCE_BROWSER YES)
set(DOXYGEN_TIMESTAMP YES)
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${HOMEPAGE_MD})
set(DOXYGEN_WARN_AS_ERROR YES)
set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)

get_target_property(GTest_gtest_INTERFACE_INCLUDE_DIRECTORIES GTest::gtest INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(Kokkos_kokkoscore_INTERFACE_INCLUDE_DIRECTORIES Kokkos::kokkoscore INTERFACE_INCLUDE_DIRECTORIES)
set(DOXYGEN_INCLUDE_PATH
    ${Kokkos_kokkoscore_INTERFACE_INCLUDE_DIRECTORIES}
    ${GTest_gtest_INTERFACE_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}
)

#---- Extract files for documentation.
get_property(GraphDispatching_FILES_FOR_DOC GLOBAL PROPERTY GraphDispatching_FILES_FOR_DOC)
list(LENGTH GraphDispatching_FILES_FOR_DOC GraphDispatching_FILES_FOR_DOC_LENGTH)
if(NOT GraphDispatching_FILES_FOR_DOC_LENGTH EQUAL 115)
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
