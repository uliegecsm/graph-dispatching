#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "Kokkos_Core.hpp"

/**
 * @addtogroup unittests
 *
 * Diamond with @c stdexec
 * -----------------------
 *
 * Create a diamond graph with @c stdexec.
 *
 * The test can be found in @ref test_diamond.stdexec.cpp.
 */

namespace tests::graph::diamond
{

/**
 * @brief Add @ref value to @ref data.
 *
 * @note The @ref offset is needed for ranges that do not start at 0.
 *       Indeed, @c stdexec::bulk currently only takes an integer for the "shape".
 */
template <typename ViewType>
struct AddValueOffset
{
    ViewType data;
    typename ViewType::value_type value;
    typename ViewType::size_type offset = 0;

    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T index) const {
        data(offset + index) += value;
    }
};

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
    view_t data("data");

    //! Define the graph. Use a simple syntax.
    constexpr int value_A = 5, value_B = 42, value_C = 156, value_D = 453;

    ::stdexec::sender auto entry = ::stdexec::just();

    ::stdexec::sender auto node_A = entry | ::stdexec::bulk(size, AddValueOffset{.data = data, .value = value_A}) | ::stdexec::split();

    ::stdexec::sender auto node_B = node_A | ::stdexec::bulk(size/2, AddValueOffset{.data = data, .value = value_B});
    ::stdexec::sender auto node_C = node_A | ::stdexec::bulk(size/2, AddValueOffset{.data = data, .value = value_C, .offset = size/2});

    ::stdexec::sender auto node_D = ::stdexec::when_all(node_B, node_C) | ::stdexec::bulk(size, AddValueOffset{.data = data, .value = value_D});

    //! Execute the graph.
    ::stdexec::sync_wait(::stdexec::start_on(pool.get_scheduler(), node_D));

    //! Check results.
    bool result = false;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<execution_space>(0, size),
        KOKKOS_LAMBDA(const auto index, bool& current) {
            const auto expt_value = value_A + (index >= size/2 ? value_C : value_B) + value_D;
            current = data(index) == expt_value;
        },
        Kokkos::LAnd<bool>(result)
    );
    ASSERT_TRUE(result);
}

} // namespace tests::graph::diamond
