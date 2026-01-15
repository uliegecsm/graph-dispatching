#include "gtest/gtest.h"

#include "kokkos_ext/impl/GraphContext.hpp"

/**
 * @addtogroup unittests
 *
 * Traits of the scheduler of @c Kokkos::Experimental::GraphContext
 * ----------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::details::graph::Scheduler is a proper scheduler.
 *
 * The tests can be found in @ref tests/kokkos_ext/graph/test_scheduler.cpp.
 *
 * References:
 *  * https://eel.is/c++draft/exec.sched
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext {

using graph_context_t = Kokkos::Experimental::GraphContext<execution_space>;
using graph_scheduler_t = decltype(std::declval<const graph_context_t&>().get_scheduler());
using graph_scheduler_env_t = Kokkos::Experimental::details::graph::SchedulerEnv<execution_space>;
using graph_schedule_sender_t = typename graph_scheduler_t::Sender;

//! @test @ref Kokkos::Experimental::details::graph::Scheduler verifies @c stdexec::scheduler.
constexpr bool test_scheduler_concept() {
    static_assert(stdexec::scheduler<graph_scheduler_t>);

    /**
     * According to https://eel.is/c++draft/exec.sched#1, a valid scheduler must have a @c scheduler_concept
     * alias.
     * However, as of https://github.com/NVIDIA/stdexec/blob/0e9983599d0c95fca3fd11baa02564eb53fb14f6/include/stdexec/__detail/__schedulers.hpp#L74,
     * it is not checked by @c stdexec::scheduler.
     *
     * Related to https://github.com/NVIDIA/stdexec/issues/1406.
     */
    static_assert(std::derived_from<typename graph_scheduler_t::scheduler_concept, stdexec::scheduler_t>);

    //! According to https://eel.is/c++draft/exec.sched#1, a @c schedule invocation must return a sender.
    static_assert(stdexec::sender<decltype(stdexec::schedule(std::declval<const graph_scheduler_t&>()))>);

    return true;
}
static_assert(test_scheduler_concept());

/**
 * @test Check that querying the completion scheduler from a schedule sender of a scheduler returns the scheduler.
 * 
 * See https://eel.is/c++draft/exec.sched#5.
 */
TEST(Scheduler, round_trip_property) {
    const graph_context_t ctx{execution_space{}};
    const graph_scheduler_t sch = ctx.get_scheduler();
    ASSERT_EQ(stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(stdexec::schedule(sch))), sch);
}

/**
 * @test Check that the @c schedule method of @ref Kokkos::Experimental::details::graph::Scheduler returns
 *       @ref Kokkos::Experimental::details::graph::Scheduler::Sender.
 */
constexpr bool test_scheduler_schedule() {
    static_assert(
        std::same_as<decltype(stdexec::schedule(std::declval<const graph_scheduler_t&>())), graph_schedule_sender_t>);

    return true;
}

/**
 * @test Check that the @c stdexec::get_env query on @ref Kokkos::Experimental::details::graph::Scheduler::Sender returns
 *       @ref Kokkos::Experimental::details::graph::SchedulerEnv.
 */
constexpr bool test_schedule_sender_env() {
    static_assert(std::same_as<stdexec::env_of_t<graph_schedule_sender_t>, const graph_scheduler_env_t&>);

    return true;
}
static_assert(test_schedule_sender_env());

//! @test Check @ref Kokkos::Experimental::details::graph::Scheduler queries.
constexpr bool test_scheduler_queries() {
    static_assert(std::same_as<
                  stdexec::__query_result_t<graph_scheduler_t, stdexec::get_completion_domain_t<stdexec::set_value_t>>,
                  Kokkos::Experimental::details::graph::Domain
    >);
    static_assert(
        std::same_as<
            stdexec::__query_result_t<graph_scheduler_t, stdexec::get_completion_scheduler_t<stdexec::set_value_t>>,
            graph_scheduler_t
        >);

    return true;
}
static_assert(test_scheduler_queries());

/**
 * @test Check that the @c stdexec::get_env query on @ref Kokkos::Experimental::details::graph::Scheduler::Sender returns
 *       @ref Kokkos::Experimental::details::graph::SchedulerEnv.
 */
constexpr bool test_scheduler_sender_get_env() {
    static_assert(std::same_as<stdexec::env_of_t<typename graph_scheduler_t::Sender>, const graph_scheduler_env_t&>);

    return true;
}
static_assert(test_scheduler_sender_get_env());

//! @test Check queries of @ref Kokkos::Experimental::details::graph::SchedulerEnv.
constexpr bool test_scheduler_env_queries() {
    static_assert(
        std::same_as<
            stdexec::__query_result_t<graph_scheduler_t, stdexec::get_completion_scheduler_t<stdexec::set_value_t>>,
            graph_scheduler_t
        >);

    static_assert(std::same_as<
                  stdexec::__query_result_t<graph_scheduler_t, stdexec::get_completion_domain_t<stdexec::set_value_t>>,
                  Kokkos::Experimental::details::graph::Domain
    >);

    return true;
}
static_assert(test_scheduler_env_queries());

} // namespace tests::kokkos_ext
