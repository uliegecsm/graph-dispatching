set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/Helpers.hpp)

set(GTEST_REPEAT TARGET_ARGS --gtest_repeat=5)

add_one_test(NAME graph        ADDITIONAL_LINK_LIBRARIES KokkosKernels::kokkoskernels ${GTEST_REPEAT})
add_one_test(NAME queue        ADDITIONAL_LINK_LIBRARIES KokkosKernels::kokkoskernels ${GTEST_REPEAT})
add_one_test(NAME queue_no_tpl ADDITIONAL_LINK_LIBRARIES KokkosKernels::kokkoskernels ${GTEST_REPEAT})
