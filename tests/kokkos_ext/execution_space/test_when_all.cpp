#include "tests/Functors.hpp"
#include "tests/kokkos_ext/Helpers.hpp"
#include "tests/kokkos_ext/execution_space/Helpers.hpp"
#include "tests/stdexec/Utils.hpp"
#include "tests/utils/ThrowsWhenCopied.hpp"

/**
 * @addtogroup unittests
 *
 * Behavior of @c when_all with @c Kokkos::Experimental::ExecutionSpaceContext
 * ---------------------------------------------------------------------------
 *
 * This group of tests check the behavior of @c stdexec::when_all with
 * @ref Kokkos::Experimental::ExecutionSpaceContext.
 *
 * The tests can be found in @ref tests/kokkos_ext/execution_space/test_when_all.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext {

class WhenAllTest : public impl::ExecutionSpaceContextTest<execution_space> { };

template <char ID>
struct Waiter {
    int* counter;
    int* value;

    KOKKOS_FUNCTION
    void operator()() const noexcept {
        *value = Kokkos::atomic_fetch_add(counter, 1);
    }
};

/**
 * @test A @c stdexec::when_all with three branches, all using a unique execution space instance.
 *
 * It is expected that *e.g.* for CUDA, it would allow for the kernels in different branches to overlap.
 */
TEST_F(WhenAllTest, concurrent_branches) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const auto [exec_A, exec_B, exec_C] = Kokkos::Experimental::partition_space(exec, 1, 1, 1);

    const context_t esc_A{exec_A}, esc_B{exec_B}, esc_C{exec_C};

    Kokkos::View<int, Kokkos::SharedSpace> counter("counter");
    Kokkos::View<int[3], Kokkos::SharedSpace> values("values");

    // use continues on should use the scheduler sender
    // thgouh we could stdexec::schedule directly it tests this hypothesis
    auto sndr = ::stdexec::when_all(
        ::stdexec::just() | ::stdexec::continues_on(esc_A.get_scheduler())
            | ::stdexec::then(Waiter<'A'>{.counter = counter.data(), .value = &values(0)}),
        ::stdexec::just() | ::stdexec::continues_on(esc_B.get_scheduler())
            | ::stdexec::then(Waiter<'B'>{.counter = counter.data(), .value = &values(1)}),
        ::stdexec::just() | ::stdexec::continues_on(esc_C.get_scheduler())
            | ::stdexec::then(Waiter<'C'>{.counter = counter.data(), .value = &values(2)}));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ::stdexec::sync_wait(std::move(sndr));

    // Check with kk tools would that prevent overlap ? We should proably check in the log messages
    // of then customization which thread ID is launching the kernels. Each branch should be using its own
    // thread, otherwise we are serializing the branches.

    ASSERT_EQ(counter(), 3);

    std::cout << "A: " << values(0) << std::endl;
    std::cout << "B: " << values(1) << std::endl;
    std::cout << "C: " << values(2) << std::endl;
}

} // namespace tests::kokkos_ext