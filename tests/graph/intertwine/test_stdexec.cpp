#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "Kokkos_Core.hpp"

#include "tests/graph/diamond/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Intertwining with @c stdexec
 * ----------------------------
 *
 * Create an intertwining graph with @c stdexec, inspired by the diamond case (see @ref diamond/test_stdexec.cpp).
 *
 * The test can be found in @ref intertwine/test_stdexec.cpp.
 */

namespace tests::graph::intertwine
{

/**
 * @brief Faking an external library call that adds its workloads to the chain.
 *
 * It adds 2 asynchronous workloads. The input sender is split, and both workloads are
 * joined and returned.
 */
template <typename Sender, typename ViewType>
decltype(auto) library(Sender&& input, ViewType data)
{
    auto continued = std::forward<Sender>(input) | ::stdexec::split();

    const auto half = data.size() / 2;

    ::stdexec::sender auto one = continued | ::stdexec::bulk(
        ::stdexec::par, half,
        diamond::AddValueOffset{.data = data, .value = diamond::Values::value_B});

    ::stdexec::sender auto two = continued | ::stdexec::bulk(
        ::stdexec::par, half,
        diamond::AddValueOffset{.data = std::move(data), .value = diamond::Values::value_C, .offset = half});

    return ::stdexec::when_all(std::move(one), std::move(two));
}

//! @test Intertwine graph using @c stdexec.
TEST(graph, intertwine_stdexec)
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
        diamond::AddValueOffset{.data = data, .value = diamond::Values::value_A});

    ::stdexec::sender auto from_library = library(node_A, data);

    ::stdexec::sender auto node_D = from_library | ::stdexec::bulk(
        ::stdexec::par, size,
        diamond::AddValueOffset{.data = data, .value = diamond::Values::value_D});

    //! Execute the graph and check results.
    ::stdexec::sync_wait(::stdexec::starts_on(pool.get_scheduler(), node_D));

    ASSERT_TRUE(diamond::check_data(execution_space{}, data));
}

} // namespace tests::graph::intertwine
