#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-result")
#include "exec/repeat_until.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/kokkos_ext/Helpers.hpp"
#include "tests/kokkos_ext/graph/Helpers.hpp"
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Resubmit a graph built with @c Kokkos::Experimental::GraphContext many times
 * ----------------------------------------------------------------------------
 *
 * From https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#design-sender-consumers:
 *
 *  A sender consumer is an algorithm that takes one or more senders, which it may @c execution::connect, as parameters, and does not return a sender.
 *
 * Building a chain of senders that uses @ref Kokkos::Experimental::GraphContext will build the underlying @c Kokkos::Experimental::Graph
 * when connected.
 *
 * If the created graph cannot be "created once, submitted many times", it may not be efficient enough to compensate/amortize the graph creation
 * and instantiation cost.
 *
 * To support this primary use case (submit the same graph many times), and avoid the creation of new sender patterns, it's envisioned to use a combination
 * of @c stdexec::sync_wait and @c exec::repeat_until with an underlying shared state that manages the graph and creates it only once, despite the
 * reconnection induced by @c exec::repeat_until.
 *
 * The tests can be found in @ref tests/kokkos_ext/graph/test_resubmit.cpp.
 * 
 * todo redo doc
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext {

using namespace Kokkos::utils::callbacks;

class ResubmitTest
    : public impl::GraphContextTest<execution_space>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t =
        RecorderListener<BeginFenceEvent, BeginParallelForEvent, AllocateDataEvent, DeallocateDataEvent, ProfileEvent>;
};

/**
 * @test Check that @ref Kokkos::Experimental::GraphContext leads to the graph being created, instantiated and submitted at each iteration
 *       of @c exec::repeat_effect_until.
 */
TEST_F(ResubmitTest, created_instantiated_submitted_at_each_iteration) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto chain = ::stdexec::just() | ADD_THEN | ADD_THEN
               | ::stdexec::bulk(::stdexec::par, 3, BulkFunctor{.data = data});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([&data, &esc, chain = std::move(chain)]() mutable {
            unsigned short int guard = 0;
            ::stdexec::sync_wait(
                ::exec::repeat_until(
                    ::stdexec::starts_on(esc.get_scheduler(), std::move(chain))
                    | ::stdexec::continues_on(::stdexec::inline_scheduler{}) | ::stdexec::then([&]() -> bool {
                          std::cout << "Hi from convergence check. Counter is " << data() << '.' << std::endl;
                          return (++guard) >= 3;
                      })));
        }),
        ::testing::ElementsAre(
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
            KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from")),
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
            KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from")),
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
            KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from"))));

    ASSERT_EQ(data(), (2 + 3) * 3);
}

} // namespace tests::kokkos_ext
