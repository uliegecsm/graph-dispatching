#ifndef GRAPH_DISPATCHING_TESTS_FUNCTORS_HPP
#define GRAPH_DISPATCHING_TESTS_FUNCTORS_HPP

namespace tests
{
template <typename ViewType>
struct ThenFunctor
{
    ViewType data;

    KOKKOS_FUNCTION
    void operator()() const { ++data(); }
};
} // namespace tests

#endif // GRAPH_DISPATCHING_TESTS_FUNCTORS_HPP
