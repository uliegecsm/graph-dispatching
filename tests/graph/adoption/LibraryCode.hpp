#ifndef GRAPH_DISPATCHING_TESTS_GRAPH_ADOPTION_LIBRARY_CODE_HPP
#define GRAPH_DISPATCHING_TESTS_GRAPH_ADOPTION_LIBRARY_CODE_HPP

#include "tests/IgnoreWarnings.hpp"

#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

namespace tests::graph::adoption
{

template <typename... Args>
struct Greetings
{
    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T) const {
        printf("Greetings from %s.\n", __PRETTY_FUNCTION__);
    }
};

/**
 * @brief Library code that is generic, whatever the type of @p input.
 *
 * The code should produce a performant code, single source, whether we use
 * @c Kokkos::Graph or not.
 *
 * @todo For non-graph-like @p input, this code currently submits both kernels to the same
 *       execution space instance, instead of providing asynchronous behavior.
 */
template <typename Sender>
decltype(auto) library(Sender&& input)
{
    //! @todo We need to expose our @c operator|, otherwise the compiler can't find a match.
    using Kokkos::Experimental::graph::details::operator|;

    using policy_t = Kokkos::RangePolicy<typename std::remove_reference_t<Sender>::execution_space>;

    PRAGMA_DIAGNOSTIC_PUSH
    PRAGMA_DIAGNOSTIC_IGNORED_DANGLING_REFERENCE

    decltype(auto) continued = std::forward<Sender>(input) | Kokkos::Experimental::graph::split();

    decltype(auto) one = continued | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, 1),
        Greetings<decltype(continued)>{});

    decltype(auto) two = std::forward<decltype(continued)>(continued) | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, 1),
        Greetings<decltype(continued)>{});

    PRAGMA_DIAGNOSTIC_POP

    return Kokkos::Experimental::when_all(std::forward<decltype(one)>(one), std::forward<decltype(two)>(two));
}

} // namespace tests::graph::adoption

#endif // GRAPH_DISPATCHING_TESTS_GRAPH_ADOPTION_LIBRARY_CODE_HPP
