#ifndef GRAPH_DISPATCHING_TESTS_GRAPH_ADOPTION_USERCODE_HPP
#define GRAPH_DISPATCHING_TESTS_GRAPH_ADOPTION_USERCODE_HPP

#include <mutex>
#include <typeindex>

#include "impl/Kokkos_Stacktrace.hpp"

#include "tests/graph/adoption/LibraryCode.hpp"

namespace tests::graph::adoption
{

/**
 * @brief User code that is generic, whatever the type of @p input.
 *
 * The code should produce a performant code, single source, whether we use
 * @c Kokkos::Graph or not.
 *
 * This code reproduces the diamond-like graph.
 *
 * @todo The @p Exec template argument should not be needed.
 */
template <typename Exec, typename Scheduler, typename Sender>
decltype(auto) user_code(Scheduler&& sch, Sender&& input)
{
    std::cout << "Executing the user routine with a chain starter of type:"
              << std::endl << '\t'
              << Kokkos::Impl::demangle(typeid(decltype(input)).name())
              << std::endl;

    using policy_t = Kokkos::RangePolicy<Exec>;

    PRAGMA_DIAGNOSTIC_PUSH
    PRAGMA_DIAGNOSTIC_IGNORED_DANGLING_REFERENCE

    decltype(auto) node_A = std::forward<Sender>(input) | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, 1),
        Greetings<Sender>{}
    );

    decltype(auto) from_library = library<Exec>(std::move(node_A)) | Kokkos::Experimental::graph::continues_on(std::forward<Scheduler>(sch));

    decltype(auto) node_D = std::move(from_library) | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, 1),
        Greetings<decltype(from_library)>{});

    PRAGMA_DIAGNOSTIC_POP

    return node_D;
}

} // namespace tests::graph::adoption

#endif // GRAPH_DISPATCHING_TESTS_GRAPH_ADOPTION_USERCODE_HPP
