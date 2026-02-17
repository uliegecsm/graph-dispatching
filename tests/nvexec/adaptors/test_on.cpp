#include "tests/IgnoreWarnings.hpp"
#include "gtest/gtest.h"

PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsign-compare")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
#include "exec/static_thread_pool.hpp"
#include "nvexec/stream_context.cuh"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/Utils.hpp"

namespace tests::nvexec::adaptors {

class OnTest
    : public ::testing::Test
    , public ::utils::StaticThreadPool<'A', 'B'> {
   protected:
    static constexpr size_t index_of_A = index_of<'A'>();
    static constexpr size_t index_of_B = index_of<'B'>();
    ::nvexec::stream_context stream_ctx{};
};

//! @test @c stdexec::on can be used nested, and the inner most one wins.
TEST_F(OnTest, on_nested) {
    ::stdexec::scheduler auto scheduler_A = this->pools.at(index_of_A).get_scheduler();
    ::stdexec::scheduler auto scheduler_B = this->pools.at(index_of_B).get_scheduler();
    ::stdexec::scheduler auto scheduler_C = this->stream_ctx.get_scheduler();

    std::thread::id tid;

    auto chain = ::stdexec::just()
               | ::stdexec::on(scheduler_A, ::stdexec::on(scheduler_C, ::stdexec::on(scheduler_B, THEN_STORE_ID(tid))));
    ::stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)

    ASSERT_EQ(tid, threads.at(index_of_B));
}

} // namespace tests::nvexec::adaptors
