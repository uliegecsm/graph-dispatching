#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/Utils.hpp"
#include "tests/kokkos_ext/execution_space/Helpers.hpp"
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c split by @c Kokkos::Experimental::ExecutionSpaceContext
 * ---------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::ExecutionSpaceContext properly customizes
 * @c split.
 *
 * The tests can be found in @ref tests/kokkos_ext/execution_space/test_split.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext
{

using namespace Kokkos::utils::callbacks;

class SplitTest : public impl::ExecutionSpaceContextTest<execution_space>,
                  public Kokkos::utils::tests::scoped::callbacks::Manager
{
public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent>;
    using variant_t           = std::variant    <BeginFenceEvent, BeginParallelForEvent>;
};

//! @test Use @c split and @c sync_wait right after.
TEST_F(SplitTest, split_and_sync_wait)
{
    const context_t esc{exec};

    ::stdexec::sender auto chain = ::stdexec::schedule(esc.get_scheduler())
        | ::stdexec::split();

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)] () mutable { ::stdexec::sync_wait(std::move(chain)); }),
        ::testing::IsEmpty()
    );
}

//! @test @ref Kokkos::Experimental::ExecutionSpaceContext
TEST_F(SplitTest, within)
{
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    ::stdexec::sender auto fork = ::stdexec::schedule(::stdexec::inline_scheduler{})
        | ::stdexec::split();

    auto chain = ::stdexec::when_all(
                  fork  | ::stdexec::continues_on(esc.get_scheduler()) | ADD_THEN | ADD_THEN,
        std::move(fork) | ::stdexec::continues_on(esc.get_scheduler()) | ADD_THEN | ADD_THEN
    ) | ::stdexec::then([]{ std::cout << "running after when_all on default scheduler" << std::endl; });

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)] () mutable { ::stdexec::sync_wait(std::move(chain)); }),
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "sync_wait"))
    ));

    ASSERT_EQ(data(), 1);
}

} // namespace tests::kokkos_ext
