include(${CMAKE_SOURCE_DIR}/cmake/modules/BuildArborX.cmake)

add_one_test(NAME intersection LINK_LIBRARIES ArborX::ArborX GTest::gmock)

add_subdirectory(BoundingVolumeHierarchy)
