#include "gtest/gtest.h"

#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

#include "examples/kokkos-execution/diamond.hpp"
#include "tests/Functors.hpp"
#include "tests/graph/runtime/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Runtime graph with P2300-flavored @c Kokkos
 * -------------------------------------------
 *
 * Create an runtime graph with @c Kokkos *à la* P2300.
 * By runtime graph, it is meant that some nodes might actually be removed (or rather not added)
 * from the graph at runtime based on some random heuristic, and the graph is therefore not fully
 * known at compile time.
 *
 * The test can be found in @ref runtime/test_outlook.cpp.
 */

namespace tests::graph::runtime
{

namespace diamond = examples::KokkosExecution::diamond;

DEFINE_TEST_SUITE

//! @test Runtime graph using P2300-flavored @c Kokkos.
TEST_P(GraphTest, runtime_outlook)
{
    //! @todo We need to expose our @c operator|, otherwise the compiler can't find a match.
    using Kokkos::Experimental::graph::details::operator|;

    //! Use @c Kokkos::DefaultExecutionSpace because we can synchronize in this setup.
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space    = typename execution_space::memory_space;

    //! Definition of the data type.
    constexpr size_t size = 4;

    using view_t = Kokkos::View<int[size], memory_space>;

    //! Which nodes will be added.
    const auto add_B = this->GetParam().at(0);
    const auto add_C = this->GetParam().at(1);

    //! Get some execution context.
    const execution_space exec {};

    //! Initialize the data.
    const view_t data(Kokkos::view_alloc(exec, "data"));

    //! Indices wherein each functor places its value.
    constexpr size_t index_A = 0, index_B = 1, index_C = 2, index_D = 3;

    //! Define the graph. Use a simple syntax.
    using policy_t = Kokkos::RangePolicy<execution_space>;

    auto root = Kokkos::Experimental::graph::create_graph(exec);

    auto node_A = root | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, 1),
        tests::AddValueOffset<view_t>{.data = data, .value = diamond::Values::value_A, .offset = index_A})
        | Kokkos::Experimental::graph::split();

    //! Define a type-erased sender type.
    using type_erased_sender_t = Kokkos::Experimental::GraphNodeRef<execution_space>;

    //! Placeholders for nodes B and C (needed because it cannot be default constructed).
    type_erased_sender_t for_B;
    type_erased_sender_t for_C;

    if(add_B) {
        for_B = node_A | Kokkos::Experimental::graph::parallel_for(
            policy_t(0, 1),
            tests::AddValueOffset<view_t>{.data = data, .value = diamond::Values::value_B, .offset = index_B});
    } else {
        for_B = node_A;
    }

    if(add_C) {
        for_C = node_A | Kokkos::Experimental::graph::parallel_for(
            policy_t(0, 1),
            tests::AddValueOffset<view_t>{.data = data, .value = diamond::Values::value_C, .offset = index_C});
    } else {
        for_C = node_A;
    }

    auto node_D = Kokkos::Experimental::when_all(for_B, for_C) | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, 1),
        tests::AddValueOffset<view_t>{.data = data, .value = diamond::Values::value_D, .offset = index_D});

    //! Execute the graph and check results.
    Kokkos::Experimental::graph::submit(exec, node_D);
    exec.fence();

    const auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data);

    ASSERT_IT_WENT_FINE(mirror)
}

INSTANTIATE_TEST_SUITE

} // namespace tests::graph::runtime
