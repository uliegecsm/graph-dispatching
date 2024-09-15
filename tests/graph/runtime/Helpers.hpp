#ifndef GRAPH_DISPATCHING_TESTS_GRAPH_RUNTIME_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_GRAPH_RUNTIME_HELPERS_HPP

#include <random>

namespace tests::graph::runtime
{

//! Generate randomness.
template <typename T = bool, typename U = int, U lower = 0, U upper = 1>
struct UniformDistribution
{
    template <typename Rng>
    T operator()(Rng& rng) {
        return std::uniform_int_distribution<U>{lower, upper}(rng);
    }
};

#define ASSERT_IT_WENT_FINE(data)                                   \
    ASSERT_EQ(data(index_A), diamond::Values::value_A);             \
    ASSERT_EQ(data(index_B), add_B ? diamond::Values::value_B : 0); \
    ASSERT_EQ(data(index_C), add_C ? diamond::Values::value_C : 0); \
    ASSERT_EQ(data(index_D), diamond::Values::value_D);

} // namespace tests::graph::runtime

#endif // GRAPH_DISPATCHING_TESTS_GRAPH_RUNTIME_HELPERS_HPP
