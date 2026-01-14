#include "kokkos-utils/callbacks/RecorderListener.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/kokkos_ext/Helpers.hpp"
#include "tests/kokkos_ext/graph/Helpers.hpp"
#include "tests/utils/ThrowsWhenCopied.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c sync_wait by @c Kokkos::Experimental::GraphContext
 * ----------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::GraphContext properly customizes
 * @c sync_wait.
 *
 * The tests can be found in @ref tests/kokkos_ext/graph/test_sync_wait.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext {

using GraphContextTest = impl::GraphContextTest<execution_space>;

/**
 * @test Ensure that @c sync_wait is properly customized.
 *
 * Since the graph is empty, no synchronization is needed.
 */
TEST_F(GraphContextTest, sync_wait) {
    const context_t esc{exec};

    auto chain = ::stdexec::schedule(esc.get_scheduler());

    Kokkos::utils::callbacks::Manager::initialize();

    ASSERT_THAT(
        Kokkos::utils::callbacks::RecorderListener<Kokkos::utils::callbacks::BeginFenceEvent>::record(
            [chain = std::move(chain)]() mutable {                         // NOLINT(performance-move-const-arg)
                const auto value = ::stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)
                static_assert(std::same_as<decltype(value), const std::optional<std::tuple<>>>);
                ASSERT_TRUE(value.has_value());
            }),
        ::testing::IsEmpty());

    Kokkos::utils::callbacks::Manager::finalize();
}

//! @test Check that @ref Kokkos::Experimental::details::graph::SyncWait properly rethrows if needed.
TEST_F(GraphContextTest, rethrows) {
    const context_t esc{exec};

    auto chain = ::stdexec::schedule(esc.get_scheduler()) | ::stdexec::then(::tests::utils::ThrowsWhenCopied{});

    ASSERT_THAT(
        ::tests::utils::MutableMoveToSyncWait{.sndr = std::move(chain)},
        testing::ThrowsMessage<std::runtime_error>(testing::StrEq("ThrowsWhenCopied: Throwing in copy constructor!")));
}

} // namespace tests::kokkos_ext
