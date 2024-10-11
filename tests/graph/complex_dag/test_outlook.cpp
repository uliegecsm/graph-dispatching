#include "gtest/gtest.h"

#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

#include "tests/graph/complex_dag/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Complex DAG with P2300-flavored @c Kokkos
 * -----------------------------------------
 *
 * Create a complex DAG with @c Kokkos *à la* P2300.
 *
 * The test can be found in @ref complex_dag/test_outlook.cpp.
 */

namespace tests::graph::complex_dag
{

//! @test Complex DAG using P2300-flavored @c kokkos.
TEST(graph, complex_dag_outlook)
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
    view_t data(Kokkos::view_alloc(exec, "data"));

    //! Define the graph. Use a simple syntax.
    using policy_t = Kokkos::RangePolicy<execution_space>;

    DEFINE_VALUES
    DEFINE_INDICES

    auto root = Kokkos::Experimental::graph::create_graph(exec)
        | Kokkos::Experimental::graph::split();

    auto node_A1 = root | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, 1),
        FetchValuesAndContribute(data, index_A1, value_A1));

    auto node_A2 = root | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, 1),
        FetchValuesAndContribute(data, index_A2, value_A2));

    auto node_A3 = root | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, 1),
        FetchValuesAndContribute(data, index_A3, value_A3));

    auto node_B1 = Kokkos::Experimental::when_all(node_A1, node_A2) | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, 1),
        FetchValuesAndContribute(data, {index_A1, index_A2}, index_B1, value_B1));

    auto node_B2 = node_A1 | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, 1),
        FetchValuesAndContribute(data, {index_A1}, index_B2, value_B2));

    auto node_B3 = Kokkos::Experimental::when_all(node_A1, node_A2) | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, 1),
        FetchValuesAndContribute(data, {index_A1, index_A2}, index_B3, value_B3));

    auto node_B4 = node_A3 | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, 1),
        FetchValuesAndContribute(data, {index_A3}, index_B4, value_B4));

    auto node_C1 = Kokkos::Experimental::when_all(node_B1, node_B3) | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, 1),
        FetchValuesAndContribute(data, {index_B1, index_B3}, index_C1, value_C1));

    auto node_C2 = Kokkos::Experimental::when_all(node_B2, node_B4) | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, 1),
        FetchValuesAndContribute(data, {index_B2, index_B4}, index_C2, value_C2));

    /// Execute the graph and check results.
    /// @todo I am missing a 'when_all(C1, C2)' here.
    Kokkos::Experimental::graph::submit(exec, node_C1);

    const auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data);

    ASSERT_IT_WENT_FINE(mirror)
}

} // namespace tests::graph::diamond
