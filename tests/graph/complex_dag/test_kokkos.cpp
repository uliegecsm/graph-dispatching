#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"

#include "tests/graph/complex_dag/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Complex DAG with @c Kokkos
 * --------------------------
 *
 * Create a complex DAG with @c Kokkos.
 *
 * The test can be found in @ref complex_dag/test_kokkos.cpp.
 */

namespace tests::graph::complex_dag
{

//! @test Complex DAG using @c kokkos.
TEST(graph, complex_dag_kokkos)
{
    //! Use @c Kokkos::DefaultExecutionSpace because we can synchronize in this setup.
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space    = typename execution_space::memory_space;

    //! Definition of the data type.
    constexpr size_t size = 9;

    using view_t = Kokkos::View<int[size], memory_space>;

    //! Get some execution context.
    const execution_space exec {};

    //! Initialize the data.
    const view_t data(Kokkos::view_alloc(exec, "data"));

    //! Define the graph. Use a simple syntax.
    using policy_t = Kokkos::RangePolicy<execution_space>;

    DEFINE_VALUES
    DEFINE_INDICES

    auto graph = Kokkos::Experimental::create_graph<execution_space>(exec);

    auto root = Kokkos::Impl::GraphAccess::create_root_ref(graph);

    auto node_A1 = root.then_parallel_for(
        policy_t(0, 1),
        FetchValuesAndContribute(data, index_A1, value_A1));

    auto node_A2 = root.then_parallel_for(
        policy_t(0, 1),
        FetchValuesAndContribute(data, index_A2, value_A2));

    auto node_A3 = root.then_parallel_for(
        policy_t(0, 1),
        FetchValuesAndContribute(data, index_A3, value_A3));

    auto node_B1 = Kokkos::Experimental::when_all(node_A1, node_A2).then_parallel_for(
        policy_t(0, 1),
        FetchValuesAndContribute(data, {index_A1, index_A2}, index_B1, value_B1));

    auto node_B2 = node_A1.then_parallel_for(
        policy_t(0, 1),
        FetchValuesAndContribute(data, {index_A1}, index_B2, value_B2));

    auto node_B3 = Kokkos::Experimental::when_all(node_A1, node_A2).then_parallel_for(
        policy_t(0, 1),
        FetchValuesAndContribute(data, {index_A1, index_A2}, index_B3, value_B3));

    auto node_B4 = node_A3.then_parallel_for(
        policy_t(0, 1),
        FetchValuesAndContribute(data, {index_A3}, index_B4, value_B4));

    auto node_C1 = Kokkos::Experimental::when_all(node_B1, node_B3).then_parallel_for(
        policy_t(0, 1),
        FetchValuesAndContribute(data, {index_B1, index_B3}, index_C1, value_C1));

    auto node_C2 = Kokkos::Experimental::when_all(node_B2, node_B4).then_parallel_for(
        policy_t(0, 1),
        FetchValuesAndContribute(data, {index_B2, index_B4}, index_C2, value_C2));

    //! Execute the graph and check results.
    graph.submit(exec);

    const auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data);

    ASSERT_IT_WENT_FINE(mirror)
}

} // namespace tests::graph::complex_dag
