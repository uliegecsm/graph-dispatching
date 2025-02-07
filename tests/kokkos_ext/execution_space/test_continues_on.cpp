#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
#include "exec/static_thread_pool.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/Helpers.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"

#include "tests/kokkos_ext/execution_space/Helpers.hpp"

#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c continues_on by @c Kokkos::Experimental::ExecutionSpaceContext
 * ----------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::ExecutionSpaceContext properly customizes
 * @c continues_on.
 *
 * The tests can be found in @ref tests/kokkos_ext/execution_space/test_continues_on.cpp.
 */

using      execution_space = Kokkos::DefaultExecutionSpace;
using         memory_space = typename execution_space::memory_space;
using host_execution_space = Kokkos::DefaultHostExecutionSpace;

namespace tests::kokkos_ext
{

using namespace Kokkos::utils::callbacks;

class ContinuesOnTest : public impl::ExecutionSpaceContextTest<execution_space, host_execution_space>,
                        public Kokkos::utils::callbacks::ManagerTestFixture
{
public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent>;
    using variant_t           = std::variant    <BeginFenceEvent, BeginParallelForEvent>;
};

/**
 * @test Check that @c continues_on is properly customized (with appropriate synchronization).
 *
 * This is done by switching context at each @c then, hoping that at some point there will be a write race condition if
 * the synchronization is not properly implemented, thus failing the count test.
 */
TEST_F(ContinuesOnTest, continues_on)
{
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const host_execution_space exec_h{};

    context_h_t esc_h{exec_h};
    context_t   esc  {exec};

    auto chain_0 = ::stdexec::just()
        | ::stdexec::continues_on(esc.get_scheduler())
        | ADD_THEN;
    auto chain_1 = std::move(chain_0)
        | ::stdexec::continues_on(esc_h.get_scheduler())
        | ADD_THEN;
    auto chain_2 = std::move(chain_1)
        | ::stdexec::continues_on(esc.get_scheduler())
        | ADD_THEN;

    using chain_0_t = decltype(chain_0);
    using chain_1_t = decltype(chain_1);
    using chain_2_t = decltype(chain_2);

    static_assert(::stdexec::__has_completion_scheduler<chain_0_t, ::stdexec::set_value_t>);
    static_assert(::stdexec::__has_completion_scheduler<chain_1_t, ::stdexec::set_value_t>);
    static_assert(::stdexec::__has_completion_scheduler<chain_2_t, ::stdexec::set_value_t>);

    static_assert(::stdexec::__completes_on<chain_0_t, scheduler_t>);
    static_assert(::stdexec::__completes_on<chain_1_t, scheduler_h_t>);
    static_assert(::stdexec::__completes_on<chain_2_t, scheduler_t>);

    static_assert(tests::stdexec::has_completion_signatures<chain_0_t, ::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_value_t()>);
    static_assert(tests::stdexec::has_completion_signatures<chain_1_t, ::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>);
    static_assert(tests::stdexec::has_completion_signatures<chain_2_t, ::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_value_t()>);

    static_assert(std::same_as<::stdexec::__early_domain_of_t<chain_0_t>, scheduler_domain_t>);
    static_assert(std::same_as<::stdexec::__early_domain_of_t<chain_1_t>, scheduler_h_domain_t>);
    static_assert(std::same_as<::stdexec::__early_domain_of_t<chain_2_t>, scheduler_domain_t>);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain_2)] () mutable { ::stdexec::sync_wait(std::move(chain)); }),
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR (exec),
            MATCHER_FOR_BEGIN_FENCE(exec, continues_on),
            MATCHER_FOR_BEGIN_PFOR (exec_h),
            MATCHER_FOR_BEGIN_FENCE(exec_h, continues_on),
            MATCHER_FOR_BEGIN_PFOR (exec),
            MATCHER_FOR_BEGIN_FENCE(exec, sync_wait)
        )
    );

    ASSERT_EQ(data(), 3) << "A synchronization is missing.";
}

/**
 * @test This test builds on @ref ContinuesOnTest_then_starts_on_late_customization_Test,
 *       and adds a @c continues_on followed by a @c then.
 */
TEST_F(ContinuesOnTest, then_starts_on_continues_on_then)
{
    const view_t data(Kokkos::view_alloc("data", exec));

    context_t esc{exec};

    //! Create a chain that does not start with a schedule sender, and use @c starts_on.
    auto starts_on = ::stdexec::starts_on(
        esc.get_scheduler(),
        ::stdexec::just() | ADD_THEN
    );

    using starts_on_t = decltype(starts_on);

    /// We are not able to query the completion scheduler, and the completion signatures are both the value and error channels.
    static_assert(!::stdexec::__has_completion_scheduler<starts_on_t, ::stdexec::set_value_t>);
    static_assert(tests::stdexec::has_completion_signatures<starts_on_t, ::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_value_t()>);

    //! Use @c stdexec::continues_on.
    auto continues_on = std::move(starts_on) | ::stdexec::continues_on(esc.get_scheduler());

    using continues_on_t = std::decay_t<decltype(continues_on)>;

    /// We can query the completion scheduler, and the completion is on both the value and error channels.
    static_assert(::stdexec::__has_completion_scheduler<continues_on_t, ::stdexec::set_value_t>);
    static_assert(::stdexec::__completes_on<continues_on_t, scheduler_t>);
    static_assert(tests::stdexec::has_completion_signatures<continues_on_t, ::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>);

    //! We can query the domain, it has a **non-default early** completion domain.
    static_assert(std::same_as<
        ::stdexec::__detail::__completion_domain_of<continues_on_t>,
        scheduler_domain_t
    >);

    static_assert(std::same_as<::stdexec::__early_domain_of_t<continues_on_t>, scheduler_domain_t>);

    //! Add the first @c then.
    auto then_fst = std::move(continues_on) | ADD_THEN;

    using then_fst_t = decltype(then_fst);

    /// We can still query the completion scheduler, and the completion is still on both the value and error channels.
    static_assert(::stdexec::__has_completion_scheduler<then_fst_t, ::stdexec::set_value_t>);
    static_assert(::stdexec::__completes_on<then_fst_t, scheduler_t>);
    static_assert(tests::stdexec::has_completion_signatures<then_fst_t, ::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_value_t()>);

    /// We can still query the domain.
    static_assert(std::same_as<::stdexec::__early_domain_of_t<then_fst_t>, scheduler_domain_t>);

    //! Add the second @c then.
    auto then_snd = std::move(then_fst) | ADD_THEN;

    using then_snd_t = decltype(then_snd);

    /// We can still query the completion scheduler, and the completion is still on both the value and error channels.
    static_assert(::stdexec::__has_completion_scheduler<then_snd_t, ::stdexec::set_value_t>);
    static_assert(::stdexec::__completes_on<then_snd_t, scheduler_t>);
    static_assert(tests::stdexec::has_completion_signatures<then_snd_t, ::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>);

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(then_snd)] () mutable { ::stdexec::sync_wait(std::move(chain)); }),
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR (exec),
            MATCHER_FOR_BEGIN_PFOR (exec),
            MATCHER_FOR_BEGIN_PFOR (exec),
            MATCHER_FOR_BEGIN_FENCE(exec, sync_wait)
        )
    );

    const auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data);

    ASSERT_EQ(mirror(), 3);
}

/**
 * @test late customization of the continues on and it must have fencing required
 *
 * seems that we aren't able to get any info about the exec used by the starts on so we might need to
 * customize starts on as well ?
 */
TEST_F(ContinuesOnTest, then_continues_on_then_starts_on)
{
    const auto execs = Kokkos::Experimental::partition_space(exec, 1, 1);

    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    context_t esc_0{execs.at(0)}, esc_1{execs.at(1)};

    auto chain = ::stdexec::just()
        | ADD_THEN
        | ::stdexec::continues_on(esc_1.get_scheduler())
        | ADD_THEN;

    const auto recorded_events = recorder_listener_t::record([chain = std::move(chain), &esc_0] () mutable { ::stdexec::sync_wait(::stdexec::starts_on(esc_0.get_scheduler(), std::move(chain))); });

    for (const auto& recorded_event : recorded_events) {
        std::visit([] (const auto& arg) { std::cout << "- " << arg << std::endl; }, recorded_event);
    }

    std::cout << "execs.at(0): " << Kokkos::Tools::Experimental::device_id(execs.at(0)) << std::endl;
    std::cout << "execs.at(1): " << Kokkos::Tools::Experimental::device_id(execs.at(1)) << std::endl;

    ASSERT_THAT(
        recorded_events,
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR (execs.at(0)),
            MATCHER_FOR_BEGIN_FENCE(execs.at(0), continues_on),
            MATCHER_FOR_BEGIN_PFOR (execs.at(1)),
            MATCHER_FOR_BEGIN_FENCE(execs.at(1), sync_wait)
        )
    );

    ASSERT_EQ(data(), 2);
}

/**
 * @test Check that @ref Kokkos::Experimental::ExecutionSpaceContext supports @c on, using different execution space instances
 *       of the same type.
 *
 * @todo This test should fail if there aren't any proper fencing implemented. Indeed,
 *       the workload launched with @c on uses a different execution space instance.
 */
TEST_F(ContinuesOnTest, on_with_mixed_execution_space_instances)
{
    const auto execs = Kokkos::Experimental::partition_space(exec, 1, 1);

    const view_t data(Kokkos::view_alloc("data", execs.at(0)));

    context_t esc_0{execs.at(0)}, esc_1{execs.at(1)};

    auto chain = ::stdexec::schedule(esc_0.get_scheduler())
        | ADD_THEN
        | ::stdexec::v2::on(esc_1.get_scheduler(), ADD_THEN)
        | ADD_THEN;

    ::stdexec::sync_wait(std::move(chain));

    const auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data);

    ASSERT_EQ(mirror(), 3);
}

/**
 * @test Check that @ref Kokkos::Experimental::ExecutionSpaceContext supports @c on, using different execution space instances
 *       of the different types.
 *
 * @todo This test should fail if there aren't any proper fencing implemented. Indeed,
 *       the workload launched with @c on uses a different execution space types.
 */
TEST_F(ContinuesOnTest, on_with_mixed_execution_space_types)
{
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    context_h_t esc_h{host_execution_space{}};
    context_t   esc  {exec};

    auto chain = ::stdexec::schedule(esc.get_scheduler())
        | ADD_THEN
        | ::stdexec::v2::on(esc_h.get_scheduler(), ADD_THEN)
        | ADD_THEN;

    ::stdexec::sync_wait(std::move(chain));

    ASSERT_EQ(data(), 3);
}

} // namespace tests::kokkos_ext
