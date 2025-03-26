#ifndef GRAPH_DISPATCHING_TESTS_FUNCTORS_HPP
#define GRAPH_DISPATCHING_TESTS_FUNCTORS_HPP

namespace tests
{
template <typename ViewType, bool MayThrow = true>
struct ThenFunctor
{
    ViewType data;

    //! @warning The @c noexcept might change how sender's completion signatures are computed. Without the @c noexcept, it might also complete on the error channel.
    KOKKOS_FUNCTION
    void operator()() const noexcept(MayThrow == false) { ++data(); }
};
} // namespace tests

#endif // GRAPH_DISPATCHING_TESTS_FUNCTORS_HPP
