#include "gtest/gtest.h"

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

//! @test Check that @c continues_on is properly customized, when the chain isn't started with a schedule sender.
TEST_F(ContinuesOnTest, no_schedule_sender_continues_on)
{
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto chain = ::stdexec::just()
        | ::stdexec::continues_on(esc.get_scheduler())
        | ADD_THEN;

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)] () mutable { ::stdexec::sync_wait(std::move(chain)); }),
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR (exec, then),
            MATCHER_FOR_BEGIN_FENCE(exec, sync_wait)
        )
    );

    ASSERT_EQ(data(), 1) << "A synchronization is missing.";
}

/**
 * @test Check that @c continues_on is properly customized (with appropriate synchronization) when using it many times.
 *
 * This is done by switching context at each @c then, hoping that at some point there will be a write race condition if
 * the synchronization is not properly implemented, thus failing the count test.
 */
TEST_F(ContinuesOnTest, continues_on_many)
{
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const host_execution_space exec_h{};

    const context_h_t esc_h{exec_h};
    const context_t   esc  {exec};

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
            MATCHER_FOR_BEGIN_PFOR (exec,   then),
            MATCHER_FOR_BEGIN_FENCE(exec,   continues_on),
            MATCHER_FOR_BEGIN_PFOR (exec_h, then),
            MATCHER_FOR_BEGIN_FENCE(exec_h, continues_on),
            MATCHER_FOR_BEGIN_PFOR (exec,   then),
            MATCHER_FOR_BEGIN_FENCE(exec,   sync_wait)
        )
    );

    ASSERT_EQ(data(), 3) << "A synchronization is missing.";
}

} // namespace tests::kokkos_ext
