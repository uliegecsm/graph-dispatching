#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
#include "exec/repeat_effect_until.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/kokkos_ext/Helpers.hpp"
#include "tests/kokkos_ext/graph/Helpers.hpp"
#include "tests/stdexec/Utils.hpp"
#include "tests/utils/LoadCheckAdd.hpp"

#include "kokkos_ext/impl/graph/box.hpp"

/**
 * @addtogroup unittests
 *
 * Put a @c Kokkos::Experimental::GraphContext in a box
 * ----------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::GraphContext properly works with
 * @ref Kokkos::Experimental::box.
 *
 * The tests can be found in @ref tests/kokkos_ext/graph/test_box.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext {

using namespace Kokkos::utils::callbacks;

class BoxTest
    : public impl::GraphContextTest<execution_space>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent, ProfileEvent>;

    static constexpr bool on_device = ::tests::utils::on_device<execution_space>();
};

//! @test @ref Kokkos::Experimental::box properly creates, instantiates and launch the graph when using @c stdexec::schedule. Our @c stdexec::sync_wait customization also kicks in.
TEST_F(BoxTest, schedule) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto chain =
        ::stdexec::schedule(esc.get_scheduler())
        | Kokkos::Experimental::box(
            ::stdexec::then(
                ::tests::utils::LoadCheckAddFunctor<int, on_device>{.prev = 0, .value = 2, .data = data.data()})
            | ::stdexec::then(
                ::tests::utils::LoadCheckAddFunctor<int, on_device>{.prev = 2, .value = 3, .data = data.data()}));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable { ::stdexec::sync_wait(std::move(chain)); }),
        ::testing::ElementsAre(
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
            KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 5);
}

//! @test @ref Kokkos::Experimental::box properly creates, instantiates and launch the graph when using @c stdexec::continues_on. Our @c stdexec::sync_wait customization also kicks in.
TEST_F(BoxTest, continues_on) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t esc{exec};

    auto chain =
        ::stdexec::just()
        | ::stdexec::then(::tests::utils::LoadCheckAddFunctor<int, false>{.prev = 0, .value = 3, .data = data.data()})
        | ::stdexec::continues_on(esc.get_scheduler())
        | Kokkos::Experimental::box(
            ::stdexec::then(
                ::tests::utils::LoadCheckAddFunctor<int, on_device>{.prev = 3, .value = 2, .data = data.data()})
            | ::stdexec::then(
                ::tests::utils::LoadCheckAddFunctor<int, on_device>{.prev = 5, .value = 4, .data = data.data()}));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable { ::stdexec::sync_wait(std::move(chain)); }),
        ::testing::ElementsAre(
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
            KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 9);
}

/**
 * @test Use @ref Kokkos::Experimental::box within a @c exec::repeat_effect_until to ensure that the graph data (functors) are not copied each iteration for nothing.
 *
 * Add a reference to a test without the box.
 */
TEST_F(BoxTest, repeat_effect_until) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t esc{exec};

    auto chain = ::stdexec::just()
               | ::stdexec::then(::tests::utils::AddFunctor<int, false>{.value = 3, .data = data.data()})
               | ::stdexec::continues_on(esc.get_scheduler())
               | Kokkos::Experimental::box(
                     ::stdexec::then(::tests::utils::AddFunctor<int, on_device>{.value = 2, .data = data.data()})
                     | ::stdexec::then(::tests::utils::AddFunctor<int, on_device>{.value = 4, .data = data.data()}));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([&data, chain = std::move(chain)]() mutable {
            unsigned short int guard = 0;
            ::stdexec::sync_wait(
                ::exec::repeat_effect_until(
                    std::move(chain)
                    | ::stdexec::continues_on(::stdexec::inline_scheduler{})
                    | ::stdexec::then([&data, &guard]() -> bool {
                          std::printf("Data is %d.\n", data());
                          return (++guard) >= 3;
                      })));
        }),
        ::testing::ElementsAre(
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
            KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from")),
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
            KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from")),
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
            KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from"))));

    ASSERT_EQ(data(), 3 * (3 + 2 + 4));
}

} // namespace tests::kokkos_ext
