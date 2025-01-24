#include "gtest/gtest.h"

#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

#include "tests/kokkos_ext/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Treat @c Kokkos execution spaces within the P2300 framework
 * -----------------------------------------------------------
 *
 * Check that we can mimic the scheduler-based programming from P2300 by
 * wrapping @c Kokkos execution spaces. It's mainly done with
 * @ref Kokkos::Experimental::ExecutionSpaceContext.
 *
 * The tests can be found in @ref kokkos_ext/test_exec_scheduler.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext
{

//! @test Check that @ref Kokkos::Experimental::ExecutionSpaceContext does its duty well.
TEST(ExecutionSpaceContext, then)
{
    using view_t = Kokkos::View<int[1], Kokkos::SharedSpace>;

    const execution_space exec {};

    view_t data(Kokkos::view_alloc("data", exec));

    Kokkos::Experimental::ExecutionSpaceContext esc{exec};

    auto chain = Kokkos::Experimental::schedule(esc.get_scheduler())
        | Kokkos::Experimental::graph::parallel_for(
            Kokkos::RangePolicy(0, 1),
            MyDummyFunctor{.data = data}
        )
        | Kokkos::Experimental::graph::parallel_for(
            Kokkos::RangePolicy(0, 1),
            MyDummyFunctor{.data = data}
        );

    Kokkos::Experimental::submit(exec, std::move(chain));

    exec.fence();

    ASSERT_EQ(data(0), 2);
}

} // namespace tests::kokkos_ext
