#ifndef GRAPH_DISPATCHING_TESTS_GRAPH_THEN_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_GRAPH_THEN_HELPERS_HPP

namespace tests::graph::then
{
template <typename ViewType>
struct ThenFunctor
{
    ViewType data;

    KOKKOS_FUNCTION
    void operator()() const { ++data(); }
};

} // namespace tests::graph::then

#endif // GRAPH_DISPATCHING_TESTS_GRAPH_THEN_HELPERS_HPP
