#ifndef GRAPH_DISPATCHING_TESTS_GRAPH_RUNTIME_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_GRAPH_RUNTIME_HELPERS_HPP

#define DEFINE_TEST_SUITE class GraphTest : public ::testing::TestWithParam<std::array<bool, 2>> {};

/// Randomizer to disable some nodes.
/// Note that as of @c nvcc 12.6, at least one of the B or C node must be
/// added. Otherwise, if none is added, it means that the @c when_all will try
/// to add twice the same dependencies (i;e. A and D would be linked by 2 edges).
#define INSTANTIATE_TEST_SUITE                 \
    INSTANTIATE_TEST_SUITE_P(                  \
        Randomize,                             \
        GraphTest,                             \
        ::testing::Values(                     \
            std::array<bool, 2>{true , false}, \
            std::array<bool, 2>{false, true},  \
            std::array<bool, 2>{true , true}   \
        )                                      \
    );

#define ASSERT_IT_WENT_FINE(data)                                   \
    ASSERT_EQ(data(index_A), diamond::Values::value_A);             \
    ASSERT_EQ(data(index_B), add_B ? diamond::Values::value_B : 0); \
    ASSERT_EQ(data(index_C), add_C ? diamond::Values::value_C : 0); \
    ASSERT_EQ(data(index_D), diamond::Values::value_D);

#endif // GRAPH_DISPATCHING_TESTS_GRAPH_RUNTIME_HELPERS_HPP
