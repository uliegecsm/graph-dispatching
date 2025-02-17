#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/stdexec/Utils.hpp"
#include "tests/Utils.hpp"

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

class StartsOnTest : public ::testing::Test
{
public:
    //! Retrieve the thread ID in each pool.
    void SetUp() override
    {
        ::stdexec::sync_wait(
            ::stdexec::schedule(pool_a.get_scheduler())
            | ::stdexec::then([&] { pool_a_thr = std::this_thread::get_id(); })
        );

        ::stdexec::sync_wait(
            ::stdexec::schedule(pool_b.get_scheduler())
            | ::stdexec::then([&] { pool_b_thr = std::this_thread::get_id(); })
        );

        ASSERT_NE(pool_a_thr, pool_b_thr);
    }
protected:
    std::thread::id          pool_a_thr, pool_b_thr;
    exec::static_thread_pool pool_a{1},  pool_b{1};
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
            size, [](const auto index, auto& data) {
                data[index] = ::utils::get_thread_id();
    });

    //! Run on pool A.
    ::stdexec::sender auto moved_to_another_A = ::stdexec::starts_on(pool_a.get_scheduler(), chain);
    const auto [result_A] = ::stdexec::sync_wait(std::move(moved_to_another_A)).value();

    //! Run on pool B.
    ::stdexec::sender auto moved_to_another_B = ::stdexec::starts_on(pool_b.get_scheduler(), chain);
    const auto [result_B] = ::stdexec::sync_wait(std::move(moved_to_another_B)).value();

    //! Since we used @c stdexec::just, we expect that each "executed chain" produced its own @c std::vector.
    ASSERT_EQ(result_A.size(), size);
    ASSERT_EQ(result_A.size(), result_B.size());
    ASSERT_NE(result_A.data(), result_B.data());

    /// Each thread pool contains a different thread (not sure why), but all the work of a given chain is executed by the same thread
    /// since the thread pool size is one.
    ASSERT_NE(result_A.at(0), result_B.at(0));

    ASSERT_NE(pool_a_thr, pool_b_thr);
    ASSERT_EQ(result_A.at(0), std::hash<std::thread::id>{}(pool_a_thr));
    ASSERT_EQ(result_B.at(0), std::hash<std::thread::id>{}(pool_b_thr));

    ASSERT_THAT(result_A, ::testing::Each(result_A.at(0)));
    ASSERT_THAT(result_B, ::testing::Each(result_B.at(0)));
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

    ::stdexec::sender auto chain = ::stdexec::schedule(pool_a.get_scheduler())
        | ::stdexec::bulk(
            size, [&](const auto index) {
                data[index] = ::utils::get_thread_id();
    });

    ::stdexec::sync_wait(::stdexec::starts_on(pool_b.get_scheduler(), std::move(chain)));

    ASSERT_THAT(data, ::testing::Each(std::hash<std::thread::id>{}(pool_a_thr)));
}

template <bool MayThrow>
struct ThenFunctorMayThrow
{
    void operator()() const noexcept(!MayThrow) { }
};

class StartsOnTraitsTest : public ::testing::Test
{
public:
    template <bool MayThrow>
    static ::stdexec::sender auto get_chain()
    {
        auto chain = ::stdexec::just() | ::stdexec::then(ThenFunctorMayThrow<MayThrow>{});

        //! The chain cannot be queried for its completion scheduler on the value channel.
        static_assert(!::tests::stdexec::has_completion_scheduler<decltype(chain), ::stdexec::set_value_t>);

        return chain;
    }

    template <::stdexec::sender Chain>
    ::stdexec::sender auto get_starts_on(Chain&& chain)
    {
        auto starts_on = ::stdexec::starts_on(pool.get_scheduler(), std::move(chain));

        //! The chain cannot be queried for its completion scheduler on the value channel.
        static_assert(!::tests::stdexec::has_completion_scheduler<decltype(starts_on), ::stdexec::set_value_t>);

        return starts_on;
    }

protected:
    exec::static_thread_pool pool{1};
};

/**
 * @test Check that a chain of @ref ThenFunctorMayThrow without @c noexcept terminated by a @c starts_on
 *       will complete on all channels.
 */
TEST_F(StartsOnTraitsTest, starts_on_without_noexcept)
{
    auto chain = this->get_chain<true>();

    static_assert(has_completion_signatures<decltype(chain), ::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_value_t()>);

    auto starts_on = this->get_starts_on(std::move(chain));

    static_assert(has_completion_signatures<decltype(starts_on), ::stdexec::set_stopped_t(), ::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_value_t()>);

    ::stdexec::sync_wait(std::move(starts_on));
}

/**
 * @test Check that a chain of @ref ThenFunctorMayThrow with @c noexcept terminated by a @c starts_on
 *       will complete on all but the error channels.
 */
TEST_F(StartsOnTraitsTest, starts_on_with_noexcept)
{
    auto chain = this->get_chain<false>();

    static_assert(has_completion_signatures<decltype(chain), ::stdexec::set_value_t()>);

    auto starts_on = this->get_starts_on(std::move(chain));

    static_assert(has_completion_signatures<decltype(starts_on), ::stdexec::set_stopped_t(),::stdexec::set_value_t()>);

    ::stdexec::sync_wait(std::move(starts_on));
}

} // namespace tests::stdexec::adaptors
