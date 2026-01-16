#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/Utils.hpp"
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::starts_on
 * -------------------------------
 *
 * This group of tests check the behavior of @c stdexec::starts_on.
 *
 * The test can be found in @ref tests/stdexec/adaptors/test_starts_on.cpp.
 */

namespace tests::stdexec::adaptors
{

class StartsOnTest : public utils::StaticThreadPool<'A', 'B'>, public ::testing::Test
{
public:
    static constexpr size_t index_of_A = index_of<'A'>();
    static constexpr size_t index_of_B = index_of<'B'>();
};

/// @test Simple @c stdexec::starts_on test, that builds a chain from @c stdexec::just and starts it on two distinct schedulers.
/// @note We therefore use a @c multi-shot chain (we can consume it several times).
TEST_F(StartsOnTest, twice_with_just_a_bulk)
{
    constexpr size_t size = 4;

    /// The workload will fill a vector with the thread ID that procecesses the work item.
    /// Note that we will create the chain only once, and execute it twice (once on each scheduler).
    /// Also note that @c stdexec::just will decay-copied the received value, and in this case will pass a copy
    /// to the connected receiver.
    ::stdexec::sender auto chain = ::stdexec::just(std::vector<size_t>(size, 0))
        | ::stdexec::bulk(
            ::stdexec::par, size, [](const auto index, auto& data) {
                data[index] = ::utils::get_thread_id();
    });

    //! Run on pool A.
    ::stdexec::sender auto moved_to_another_A = ::stdexec::starts_on(pools.at(index_of_A).get_scheduler(), chain);
    const auto [result_A] = ::stdexec::sync_wait(std::move(moved_to_another_A)).value();

    //! Run on pool B.
    ::stdexec::sender auto moved_to_another_B = ::stdexec::starts_on(pools.at(index_of_B).get_scheduler(), chain);
    const auto [result_B] = ::stdexec::sync_wait(std::move(moved_to_another_B)).value();

    //! Since we used @c stdexec::just, we expect that each "executed chain" produced its own @c std::vector.
    ASSERT_EQ(result_A.size(), size);
    ASSERT_EQ(result_A.size(), result_B.size());
    ASSERT_NE(result_A.data(), result_B.data());

    /// Each thread pool contains a different thread (not sure why), but all the work of a given chain is executed by the same thread
    /// since the thread pool size is one.
    ASSERT_NE(result_A.at(0), result_B.at(0));

    ASSERT_NE(threads.at(index_of_A), threads.at(index_of_B));
    ASSERT_EQ(result_A.at(0), std::hash<std::thread::id>{}(threads.at(index_of_A)));
    ASSERT_EQ(result_B.at(0), std::hash<std::thread::id>{}(threads.at(index_of_B)));

    ASSERT_THAT(result_A, ::testing::Each(result_A.at(0)));
    ASSERT_THAT(result_B, ::testing::Each(result_B.at(0)));

    /// Let's check some traits.
    /// The chain completion domain is indeterminate.
    static_assert(std::same_as<
        ::stdexec::__completion_domain_of_t<::stdexec::set_value_t, decltype(chain)>,
        ::stdexec::indeterminate_domain<>
    >);

    //! @c starts_on completes on the passed scheduler domain.
    static_assert(std::same_as<
        decltype(::stdexec::get_completion_domain<::stdexec::set_value_t>(::stdexec::get_env(moved_to_another_A))),
        exec::_pool_::_static_thread_pool::domain
    >);
}

/**
 * @test This test shows that if a chain is started using a @c stdexec::schedule
 *       sender, then using @c stdexec::starts_on later on with another @c stdexec::schedule
 *       sender is a no-op.
 *
 * @warning We are not sure if it is expected by the spec, or if it's because the @c exec::static_thread_pool has
 *          been implemented with eager customization, see also
 *          https://github.com/NVIDIA/stdexec/blob/b888185d667f68b9a8bda5d0c81d03edf9ec3fe1/include/exec/static_thread_pool.hpp#L265.
 *          Also note that, as stated in https://github.com/cplusplus/sender-receiver/blob/a1790ddda5dcdf70f0658d0b50794649caa6c96f/P2300R9.html#L4632,
 *          resource transition **must** be explicit. This test case might therefore exploit some
 *          undefined behavior.
 */
TEST_F(StartsOnTest, B_once_after_schedule_on_A_is_a_no_op)
{
    constexpr size_t size = 4;

    /// The workload will fill a vector with the thread ID that processes the work item.
    /// We start the chain with a schedule on pool A, and then try to
    /// start it on pool B, such that we could expect that the vector will be filled
    /// with the ID of the thread in pool B. But the chain will not be taken care of by the pool B
    /// but by pool A, since we started the chain with it.
    std::vector<size_t> data(size, 0);

    ::stdexec::sender auto chain = ::stdexec::schedule(pools.at(index_of_A).get_scheduler())
        | ::stdexec::bulk(
            ::stdexec::par, size, [&](const auto index) {
                data[index] = ::utils::get_thread_id();
    });

    ::stdexec::sync_wait(::stdexec::starts_on(pools.at(index_of_B).get_scheduler(), std::move(chain))); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(data, ::testing::Each(std::hash<std::thread::id>{}(threads.at(index_of_A))));
}

//! @test Passing a chain that is still in the default domain is fully started on the scheduler of @c starts_on.
TEST_F(StartsOnTest, starts_on_goes_to_begin_of_chain)
{
    std::array<std::thread::id, 3> ids;

    auto chain = ::stdexec::just()
        | ::stdexec::then([&ids]{ ids[0] = std::this_thread::get_id(); })
        | ::stdexec::then([&ids]{ ids[1] = std::this_thread::get_id(); });

    static_assert(std::same_as<
        ::stdexec::__completion_domain_of_t<::stdexec::set_value_t, decltype(chain)>,
        ::stdexec::indeterminate_domain<>
    >);

    auto work = ::stdexec::starts_on(pools.at(index_of_A).get_scheduler(), std::move(chain)) // NOLINT(performance-move-const-arg)
        | ::stdexec::then([&ids]{ ids[2] = std::this_thread::get_id(); });

    ::stdexec::sync_wait(std::move(work)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(ids, ::testing::Each(threads.at(index_of_A)));
}

template <bool MayThrow>
struct ThenFunctorMayThrow
{
    void operator()() const noexcept(!MayThrow) { }
};

class StartsOnTraitsTest : public utils::StaticThreadPool<'A'>, public ::testing::Test
{
public:
    template <bool MayThrow>
    static ::stdexec::sender auto get_chain() {
        return ::stdexec::just() | ::stdexec::then(ThenFunctorMayThrow<MayThrow>{});
    }

    template <::stdexec::sender Chain>
    ::stdexec::sender auto get_starts_on(Chain&& chain)
    {
        auto starts_on = ::stdexec::starts_on(pools.front().get_scheduler(), std::forward<Chain>(chain));

        //! The chain can be queried for its completion scheduler on the value channel.
        static_assert(requires {
            ::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(starts_on));
        });

        return starts_on;
    }
};

/**
 * @test Check that a chain of @ref ThenFunctorMayThrow without @c noexcept terminated by a @c starts_on
 *       will complete on all channels.
 */
TEST_F(StartsOnTraitsTest, starts_on_without_noexcept)
{
    auto chain = this->get_chain<true>();

    static_assert(has_completion_signatures<decltype(chain), ::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_value_t()>);

    auto starts_on = this->get_starts_on(std::move(chain)); // NOLINT(performance-move-const-arg)

    //! Until it is connected, the completion signatures are *dependent* (they are not fully known yet).
    using starts_on_t = decltype(starts_on);

    static_assert(std::derived_from<
        ::std::invoke_result_t<::stdexec::get_completion_signatures_t, starts_on_t>,
        ::stdexec::dependent_sender_error
    >);

    static_assert(std::same_as<std::invoke_result_t<
        ::stdexec::get_completion_signatures_t, starts_on_t, ::stdexec::env<>>,
        ::stdexec::completion_signatures<::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_value_t(), ::stdexec::set_stopped_t()>
    >);

    ::stdexec::sync_wait(std::move(starts_on)); // NOLINT(performance-move-const-arg)
}

/**
 * @test Check that a chain of @ref ThenFunctorMayThrow with @c noexcept terminated by a @c starts_on
 *       will complete on all but the error channels.
 */
TEST_F(StartsOnTraitsTest, starts_on_with_noexcept)
{
    auto chain = this->get_chain<false>();

    static_assert(has_completion_signatures<decltype(chain), ::stdexec::set_value_t()>);

    auto starts_on = this->get_starts_on(std::move(chain)); // NOLINT(performance-move-const-arg)

    //! Until it is connected, the completion signatures are *dependent* (they are not fully known yet).
    using starts_on_t = decltype(starts_on);

    static_assert(std::derived_from<
        ::std::invoke_result_t<::stdexec::get_completion_signatures_t, starts_on_t>,
        ::stdexec::dependent_sender_error
    >);

    static_assert(std::same_as<std::invoke_result_t<
        ::stdexec::get_completion_signatures_t, starts_on_t, ::stdexec::env<>>,
        ::stdexec::completion_signatures<::stdexec::set_value_t(), ::stdexec::set_stopped_t()>
    >);

    ::stdexec::sync_wait(std::move(starts_on)); // NOLINT(performance-move-const-arg)
}

} // namespace tests::stdexec::adaptors
