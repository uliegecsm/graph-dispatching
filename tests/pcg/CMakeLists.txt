set(GTEST_REPEAT TARGET_ARGS --gtest_repeat=5)

add_one_test(NAME graph ADDITIONAL_LINK_LIBRARIES KokkosKernels::kokkoskernels ${GTEST_REPEAT})
add_one_test(NAME queue ADDITIONAL_LINK_LIBRARIES KokkosKernels::kokkoskernels ${GTEST_REPEAT})
