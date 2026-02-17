#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::on
 * ------------------------
 *
 * This group of tests check the behavior of @c stdexec::on.
 *
 * The test can be found in @ref stdexec/adaptors/test_on.cpp.
 */

namespace tests::stdexec::adaptors {

class OnTest
    : public utils::StaticThreadPool<'A', 'B'>
    , public ::testing::Test {
   public:
    static constexpr size_t index_of_A = index_of<'A'>();
    static constexpr size_t index_of_B = index_of<'B'>();
};

//! @test Simple test for @c stdexec::on showing that the context transition is indeed temporary.
TEST_F(OnTest, on) {
    ::stdexec::scheduler auto scheduler_A = this->pools.at(index_of_A).get_scheduler();
    ::stdexec::scheduler auto scheduler_B = this->pools.at(index_of_B).get_scheduler();

    std::array<std::thread::id, 3> thrids;
    size_t counter = 0;

    auto chain = ::stdexec::schedule(scheduler_A) | THEN_STORE_ID(thrids[0], { ++counter; })
               | ::stdexec::on(scheduler_B, THEN_STORE_ID(thrids[1], { ++counter; }))
               | THEN_STORE_ID(thrids[2], { ++counter; });

    ::stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)

    //! The second @c then has indeed been executed by the second thread pool.
    ASSERT_THAT(thrids, ::testing::ElementsAre(threads.at(index_of_A), threads.at(index_of_B), threads.at(index_of_A)));

    ASSERT_EQ(counter, 3);
}

template <::stdexec::scheduler Schd = ::stdexec::inline_scheduler>
struct env_with_scheduler {
    constexpr auto query(::stdexec::get_scheduler_t) const noexcept {
        return Schd{};
    }
};

//! @test Completion signatures after @c stdexec::on derive from @c stdexec::dependent_sender_error.
TEST_F(OnTest, completion_signatures) {
    auto chain = ::stdexec::just() | ::stdexec::on(::stdexec::inline_scheduler{}, ::stdexec::then([] { }));

    using chain_t = decltype(chain);

    /**
     * The @c stdexec::on algorithm needs to restore execution back to
     * the original scheduler afterward.
     * Therefore, since the sender starts with @c stdexec::just,
     * trying to retrieve the completion signatures without providing an environment
     * containing a scheduler will return @c stdexec::dependent_sender_error.
     */
    static_assert(requires {
        {
            ::stdexec::get_completion_signatures(std::declval<chain_t>())
        } -> std::derived_from<::stdexec::dependent_sender_error>;
    });

    static_assert(std::same_as<
                  ::stdexec::__completion_signatures_of_t<chain_t, env_with_scheduler<>>,
                  ::stdexec::completion_signatures<::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>
    >);
}

//! @test @c stdexec::on can be used nested, and the inner most one wins.
TEST_F(OnTest, on_nested) {
    ::stdexec::scheduler auto scheduler_A = this->pools.at(index_of_A).get_scheduler();
    ::stdexec::scheduler auto scheduler_B = this->pools.at(index_of_B).get_scheduler();

    std::thread::id tid;

    auto chain = ::stdexec::just() | ::stdexec::on(scheduler_A, ::stdexec::on(scheduler_B, THEN_STORE_ID(tid)));
    ::stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)

    ASSERT_EQ(tid, threads.at(index_of_B));
}

} // namespace tests::stdexec::adaptors
