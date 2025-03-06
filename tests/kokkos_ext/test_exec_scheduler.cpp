#include "gtest/gtest.h"

#include "kokkos_ext/impl/ExecutionSpaceContext.hpp"

/**
 * @addtogroup unittests
 *
 * Treat @c Kokkos execution spaces within the P2300 framework
 * -----------------------------------------------------------
 *
 * Check that we can use the scheduler-based programming from P2300 by
 * wrapping @c Kokkos execution spaces. It's mainly done with
 * @ref Kokkos::Experimental::ExecutionSpaceContext.
 *
 * The tests can be found in @ref kokkos_ext/test_exec_scheduler.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext
{

struct ExecutionSpaceContextTest : public ::testing::Test
{
public:
    using context_t         = Kokkos::Experimental::ExecutionSpaceContext<execution_space>;
    using scheduler_t       = decltype(std::declval<const context_t>().get_scheduler());
    using schedule_sender_t = decltype(::stdexec::schedule(std::declval<scheduler_t>()));

public:
    static void SetUpTestSuite() {
        exec = Kokkos::Experimental::partition_space(execution_space{}, 1)[0];
    }

    static void TearDownTestSuite() { exec.reset(); }

protected:
    static inline std::optional<execution_space> exec = std::nullopt;
};

/// @test Check that @ref Kokkos::Experimental::ExecutionSpaceContext
///       satisfies the @c stdexec::scheduler concept.
///       Check that it has a valid schedule sender.
TEST_F(ExecutionSpaceContextTest, is_a_scheduler)
{
    static_assert(::stdexec::sender   <schedule_sender_t>);
    static_assert(::stdexec::scheduler<scheduler_t>);

    const context_t context{*exec};

    const stdexec::scheduler auto sch = context.get_scheduler();

    stdexec::sender auto sndr = stdexec::schedule(sch);
}

} // namespace tests::kokkos_ext
