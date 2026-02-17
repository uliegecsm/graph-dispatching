#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
#include "exec/split.hpp"
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "Kokkos_Core.hpp"

#include "tests/graph/complex_dag/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Complex DAG with @c stdexec
 * ---------------------------
 *
 * Create a complex DAG with @c stdexec.
 *
 * The test can be found in @ref complex_dag/test_stdexec.cpp.
 */

namespace tests::graph::complex_dag
{

//! @test Complex DAG using @c stdexec.
TEST(graph, complex_dag_stdexec)
{
    //! Use @c Kokkos::Serial because we have no mean to synchronize in this setup.
    using execution_space = Kokkos::Serial;
    using memory_space    = typename execution_space::memory_space;

    //! Definition of the data type.
    constexpr size_t size = 9;

    using view_t = Kokkos::View<int[size], memory_space>;

    //! Get some execution context.
    ::exec::static_thread_pool pool{1};

    //! Initialize the data.
    const view_t data(Kokkos::view_alloc("data"));

    //! Define the graph. Use a simple syntax.
    DEFINE_VALUES
    DEFINE_INDICES

    ::stdexec::sender auto entry = ::stdexec::just() | ::exec::split();

    ::stdexec::sender auto node_A1 = entry | ::stdexec::bulk(
        ::stdexec::par, 1,
        FetchValuesAndContribute(data, index_A1, value_A1))
        | ::exec::split();

    ::stdexec::sender auto node_A2 = entry | ::stdexec::bulk(
        ::stdexec::par, 1,
        FetchValuesAndContribute(data, index_A2, value_A2))
        | ::exec::split();

    ::stdexec::sender auto node_A3 = entry | ::stdexec::bulk(
        ::stdexec::par, 1,
        FetchValuesAndContribute(data, index_A3, value_A3));

    ::stdexec::sender auto node_B1 = ::stdexec::when_all(node_A1, node_A2) | ::stdexec::bulk(
        ::stdexec::par, 1,
        FetchValuesAndContribute(data, {index_A1, index_A2}, index_B1, value_B1));

    ::stdexec::sender auto node_B2 = node_A1 | ::stdexec::bulk(
        ::stdexec::par, 1,
        FetchValuesAndContribute(data, {index_A1}, index_B2, value_B2));

    ::stdexec::sender auto node_B3 = ::stdexec::when_all(node_A1, node_A2) | ::stdexec::bulk(
        ::stdexec::par, 1,
        FetchValuesAndContribute(data, {index_A1, index_A2}, index_B3, value_B3));

    ::stdexec::sender auto node_B4 = node_A3 | ::stdexec::bulk(
        ::stdexec::par, 1,
        FetchValuesAndContribute(data, {index_A3}, index_B4, value_B4));

    ::stdexec::sender auto node_C1 = ::stdexec::when_all(node_B1, node_B3) | ::stdexec::bulk(
        ::stdexec::par, 1,
        FetchValuesAndContribute(data, {index_B1, index_B3}, index_C1, value_C1));

    ::stdexec::sender auto node_C2 = ::stdexec::when_all(node_B2, node_B4) | ::stdexec::bulk(
        ::stdexec::par, 1,
        FetchValuesAndContribute(data, {index_B2, index_B4}, index_C2, value_C2));

    //! Execute the graph and check results.
    ::stdexec::sync_wait(::stdexec::starts_on(pool.get_scheduler(), ::stdexec::when_all(node_C1, node_C2)));

    ASSERT_IT_WENT_FINE(data)
}

} // namespace tests::graph::complex_dag
