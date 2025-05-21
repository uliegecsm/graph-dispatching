#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "Kokkos_Core.hpp"

#include "tests/graph/diamond/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Diamond with @c stdexec
 * -----------------------
 *
 * Create a diamond graph with @c stdexec.
 *
 * The test can be found in @ref diamond/test_stdexec.cpp.
 */

namespace tests::graph::diamond
{

//! @test Diamond graph using @c stdexec.
TEST(graph, diamond_stdexec)
{
    //! Use @c Kokkos::Serial because we have no mean to synchronize in this setup.
    using execution_space = Kokkos::Serial;
    using memory_space    = typename execution_space::memory_space;

    //! Definition of the data type.
    constexpr size_t size = 10;
    static_assert(size % 2 == 0);

    using view_t = Kokkos::View<int[size], memory_space>;

    //! Get some execution context.
    ::exec::static_thread_pool pool{1};

    //! Initialize the data.
    const view_t data(Kokkos::view_alloc("data"));

    //! Define the graph. Use a simple syntax.
    ::stdexec::sender auto entry = ::stdexec::just();

    ::stdexec::sender auto node_A = entry | ::stdexec::bulk(
        ::stdexec::par, size,
        AddValueOffset{.data = data, .value = Values::value_A})
        | ::stdexec::split();

    ::stdexec::sender auto node_B = node_A | ::stdexec::bulk(
        ::stdexec::par, size / 2,
        AddValueOffset{.data = data, .value = Values::value_B});
    ::stdexec::sender auto node_C = node_A | ::stdexec::bulk(
        ::stdexec::par, size / 2,
        AddValueOffset{.data = data, .value = Values::value_C, .offset = size / 2});

    ::stdexec::sender auto node_D = ::stdexec::when_all(node_B, node_C) | ::stdexec::bulk(
        ::stdexec::par, size,
        AddValueOffset{.data = data, .value = Values::value_D});

    //! Execute the graph and check results.
    ::stdexec::sync_wait(::stdexec::starts_on(pool.get_scheduler(), node_D));

    ASSERT_TRUE(check_data(execution_space{}, data));
}

} // namespace tests::graph::diamond
