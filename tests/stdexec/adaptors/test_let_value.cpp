#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/Utils.hpp"
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::let_value
 * -------------------------------
 *
 * This group of tests check the behavior of @c stdexec::let_value.
 *
 * The test can be found in @ref stdexec/adaptors/test_let_value.cpp.
 */

namespace tests::stdexec::adaptors {

class LetValueTest
    : public utils::StaticThreadPool<'A', 'B', 'C', 'D'>
    , public ::testing::Test {
   public:
    static constexpr size_t index_of_A = index_of<'A'>();
    static constexpr size_t index_of_B = index_of<'B'>();
    static constexpr size_t index_of_C = index_of<'C'>();
    static constexpr size_t index_of_D = index_of<'D'>();
};

/**
 * @test Use @c stdexec::let_value to express branching instead of @c stdexec::split, as proposed in @ref P3682R0.
 *
 * The semantic seems different. With @c stdexec::let_value, the inner sender is connected "lately", only when the @c stdexec::let_value
 * receiver is consumed.
 */
TEST_F(LetValueTest, for_branching) {
    std::array<std::thread::id, 4> thrids;

    std::atomic<bool> consumed{false};

    auto shared = ::stdexec::schedule(pools.at(index_of_A).get_scheduler()) | THEN_STORE_ID(thrids[0])
                | ::stdexec::then([&consumed]() -> double {
                      std::printf("Preliminary 'then' returning.\n");
                      if (consumed)
                          throw std::runtime_error("Already consumed.");
                      consumed = true;
                      return 42.;
                  });

    ::stdexec::sync_wait(
        std::move(shared) | ::stdexec::let_value([this, &consumed, &thrids](const double value) {
            std::printf("Value received is %f.\n", value);
            if (!consumed)
                throw std::runtime_error("Not consumed yet.");
            auto br_b = ::stdexec::schedule(pools.at(index_of_B).get_scheduler()) | THEN_STORE_ID(thrids[1]);
            auto br_c = ::stdexec::schedule(pools.at(index_of_C).get_scheduler()) | THEN_STORE_ID(thrids[2]);
            auto br_d = ::stdexec::schedule(pools.at(index_of_D).get_scheduler()) | THEN_STORE_ID(thrids[3]);
            return ::stdexec::when_all(std::move(br_b), std::move(br_c), std::move(br_d));
        }));

    ASSERT_EQ(thrids[0], threads.at(index_of_A));
    ASSERT_EQ(thrids[1], threads.at(index_of_B));
    ASSERT_EQ(thrids[2], threads.at(index_of_C));
    ASSERT_EQ(thrids[3], threads.at(index_of_D));

    ASSERT_TRUE(consumed);
}

} // namespace tests::stdexec::adaptors
