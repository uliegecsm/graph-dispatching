#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
#include <stdexec/execution.hpp>
PRAGMA_DIAGNOSTIC_POP

/**
 * @addtogroup unittests
 *
 * Create a custom scheduler
 * -------------------------
 *
 * Create a simple custom scheduler that verifies the @c stdexec::scheduler concept.
 *
 * This test is heavily inspired by https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/test/stdexec/concepts/test_concept_scheduler.cpp.
 *
 * The test can be found in @ref stdexec/test_custom_scheduler.cpp.
 *
 * References:
 *  - https://en.cppreference.com/w/cpp/execution/scheduler
 *  - https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/stdexec/__detail/__schedulers.hpp#L72-L77
 */

namespace tests::stdexec
{

//! Custom scheduler.
struct MyScheduler
{
    template <class Scheduler>
    struct MyEnv
    {
        template <typename CompletionTag>
        Scheduler query(::stdexec::get_completion_scheduler_t<CompletionTag>) const noexcept { return {}; }
    };

    /**
     * @brief Operation state.
     *
     * See also https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#spec-execution.opstate.
     */
    template <class R>
    struct MyOp
    {
        using operation_state_concept = ::stdexec::operation_state_t;
        R rcvr;

        void start() & noexcept { ::stdexec::set_value(std::move(rcvr)); }
    };

    /**
     * @brief Custom sender.
     *
     * According to @c P2300.R8 (https://github.com/cplusplus/sender-receiver/blob/a1790ddda5dcdf70f0658d0b50794649caa6c96f/P2300R8.html#L3652-L3653):
     *      @c enable_sender and @c enable_receiver traits now have default implementations that look for
     *      nested @c sender_concept and @c receiver_concept types, respectively.
     *
     * We are not sure why, but it seems related to the customization points, see:
     *  - https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#design-dispatch.
     *  - https://github.com/cplusplus/sender-receiver/blob/a1790ddda5dcdf70f0658d0b50794649caa6c96f/P2300R9.html#L5928
     */
    struct MySender
    {
        using sender_concept = ::stdexec::sender_t;

        //! This sender only completes on the value channel. It passes no arguments to @c set_value.
        using completion_signatures = ::stdexec::completion_signatures<::stdexec::set_value_t()>;

        //! @note @c connect is not needed for fulfilling the @c stdexec::sender concept.
        template <::stdexec::receiver_of<completion_signatures> R>
        MyOp<std::remove_cvref_t<R>> connect(R&& rcvr) noexcept(std::is_nothrow_move_constructible_v<R>) { return {std::forward<R>(rcvr)}; }

        MyEnv<MyScheduler> get_env() const noexcept { return {}; }
    };

    MySender schedule() const noexcept { return {}; }

    friend bool operator==(const MyScheduler&, const MyScheduler&) noexcept { return true; }
};

//! @test Check @ref MyScheduler traits.
TEST(stdexec, my_scheduler_traits)
{
    static_assert(::stdexec::sender   <MyScheduler::MySender>);
    static_assert(::stdexec::scheduler<MyScheduler>);
}

//! @test Check that @ref MyScheduler can be used with @c stdexec::then.
TEST(stdexec, my_scheduler_then)
{
    unsigned short int counter = 0;

    auto chain = ::stdexec::schedule(MyScheduler{}) | ::stdexec::then([&counter](){ ++counter; });

    ::stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)

    ASSERT_EQ(counter, 1);
}

} // namespace tests::stdexec
