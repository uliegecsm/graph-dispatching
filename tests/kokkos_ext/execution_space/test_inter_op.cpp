#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
// PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
// PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
#include "exec/static_thread_pool.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/Helpers.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"

#include "tests/kokkos_ext/execution_space/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Interoperability of @c Kokkos::Experimental::ExecutionSpaceContext with other schedulers
 * ----------------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::ExecutionSpaceContext can be used in
 * conjunction with other schedulers like @c exec::static_thread_pool.
 *
 * The tests can be found in @ref kokkos_ext/execution_space/test_inter_op.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext
{

using namespace Kokkos::utils::callbacks;

class InterOpTest : public impl::ExecutionSpaceContextTest<execution_space>,
                    public Kokkos::utils::callbacks::ManagerTestFixture
{
public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent>;
    using variant_t           = std::variant    <BeginFenceEvent, BeginParallelForEvent>;
};

//! @test Check that @ref Kokkos::Experimental::ExecutionSpaceContext can be used along with @c exec::static_thread_pool.
TEST_F(InterOpTest, static_thread_pool)
{
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    exec::static_thread_pool pool{1};

    ::stdexec::scheduler auto scheduler_exec = esc .get_scheduler();
    ::stdexec::scheduler auto scheduler_pool = pool.get_scheduler();

    auto chain = ::stdexec::schedule(scheduler_exec)
        | ADD_THEN
        | ::stdexec::continues_on(scheduler_pool)
        | ADD_THEN
        | ::stdexec::continues_on(scheduler_exec)
        | ADD_THEN;

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)] () mutable {
            ::stdexec::sync_wait(std::move(chain));
        }),
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR (exec, then),
            MATCHER_FOR_BEGIN_FENCE(exec, sync_wait)
        )
    );

    ASSERT_EQ(data(), 3);
}

} // namespace tests::kokkos_ext
