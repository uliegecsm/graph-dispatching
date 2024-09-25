#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
#include "exec/static_thread_pool.hpp"
#include "exec/any_sender_of.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "Kokkos_Core.hpp"

#include "tests/graph/diamond/Helpers.hpp"
#include "tests/graph/runtime/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Runtime graph with @c stdexec
 * -----------------------------
 *
 * Create an runtime graph with @c stdexec, inspired by the diamond case (see @ref diamond/test_stdexec.cpp).
 * By runtime graph, it is meant that some nodes might actually be removed (or rather not added)
 * from the graph at runtime based on some random heuristic, and the graph is therefore not fully
 * known at compile time.
 *
 * The test can be found in @ref runtime/test_stdexec.cpp.
 */

namespace tests::graph::runtime
{

/**
 * @brief Helper for defining a "any" sender type.
 *
 * Inspired by https://github.com/NVIDIA/stdexec/blob/8bc7c7f06fe39831dea6852407ebe7f6be8fa9fd/examples/benchmark/fibonacci.cpp.
 */
template <class... Ts>
using any_sender_of =
  typename ::exec::any_receiver_ref<::stdexec::completion_signatures<Ts...>>::template any_sender<>;

/**
 * @test Runtime graph using @c stdexec.
 *
 * Weird things that we need to do (compared to pure compile time version), that is, things
 * we have to do when using "any" senders:
 *  - @c noexcept on the signature of the call operator of our functors.
 *  - We must move them.
 *  - There is no "dynamic" @c when_all, so we need to create fake empty nodes (sse below).
 */
TEST(graph, runtime_stdexec)
{
    //! Use @c Kokkos::Serial because we have no mean to synchronize in this setup.
    using execution_space = Kokkos::Serial;
    using memory_space    = typename execution_space::memory_space;

    //! Definition of the data type.
    constexpr size_t size = 4;

    using view_t = Kokkos::View<int[size], memory_space>;

    //! Randomizer to disable some nodes.
    UniformDistribution randomizer{};
    std::mt19937 gen(std::random_device{}());

    const bool add_B = randomizer(gen);
    const bool add_C = randomizer(gen);

    //! Get some execution context.
    ::exec::static_thread_pool pool{1};

    //! Initialize the data.
    view_t data(Kokkos::view_alloc("data"));

    //! Indices wherein each functor places its value.
    constexpr size_t index_A = 0, index_B = 1, index_C = 2, index_D = 3;

    //! Define the graph. Use a simple syntax.
    ::stdexec::sender auto entry = ::stdexec::just();

    ::stdexec::sender auto node_A = entry | ::stdexec::bulk(
        1,
        diamond::AddValueOffset{.data = data, .value = diamond::Values::value_A, .offset = index_A})
        | ::stdexec::split();

    //! Define a type-erased sender type. See also https://github.com/NVIDIA/stdexec/issues/1411.
    using type_erased_sender_t = any_sender_of<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr const&), stdexec::set_stopped_t()>;

    //! Placeholders for nodes B and C (needed because it cannot be default constructed).
    std::optional<type_erased_sender_t> for_B;
    std::optional<type_erased_sender_t> for_C;

    if(add_B) {
        for_B = node_A | ::stdexec::bulk(
            1,
            diamond::AddValueOffset{.data = data, .value = diamond::Values::value_B, .offset = index_B});
    } else {
        for_B = node_A;
    }

    if(add_C) {
        for_C = node_A | ::stdexec::bulk(
            1,
            diamond::AddValueOffset{.data = data, .value = diamond::Values::value_C, .offset = index_C});
    } else {
        for_C = node_A;
    }

    ::stdexec::sender auto node_D = ::stdexec::when_all(std::move(*for_B), std::move(*for_C)) | ::stdexec::bulk(
        1,
        diamond::AddValueOffset{.data = data, .value = diamond::Values::value_D, .offset = index_D});

    //! Execute the graph and check results.
    ::stdexec::sync_wait(::stdexec::start_on(pool.get_scheduler(), std::move(node_D)));

    ASSERT_IT_WENT_FINE(data)
}

} // namespace tests::graph::runtime
