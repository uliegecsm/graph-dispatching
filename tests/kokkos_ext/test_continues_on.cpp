#include "gtest/gtest.h"

#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

#include "tests/kokkos_ext/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * @c Kokkos extensions for graph-compatible parallel-for construct
 * ----------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos_Graph_Execution.hpp effectively
 * makes it possible to use a parallel-for construct in a templated code in either
 * graph or execution space instance mode transparently.
 *
 * The tests can be found in @ref kokkos_ext/test_parallel_for.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;
using memory_space    = typename execution_space::memory_space;

namespace tests::kokkos_ext
{

class ContinuesOnTest : public ::testing::Test
{
public:
    using view_t = Kokkos::View<int[1], execution_space, Kokkos::MemoryTraits<Kokkos::Atomic>>;

public:
    void SetUp() override
    {
        this->execs = Kokkos::Experimental::partition_space(execution_space{}, 1, 1);
        this->data = view_t(Kokkos::view_alloc("data", execs.at(0)));
    }
protected:
    std::vector<execution_space> execs;
    view_t data;
};

/**
 * @test Check the execution space instance mode for the parallel-for construct
 *       with many execution space instance involved.
 *
 * The test relies on a @c std::execution::continues_on like idea.
 */
TEST_F(ContinuesOnTest, many_exec_continues_on)
{
    using Kokkos::Experimental::continues_on;

    using Kokkos::Experimental::graph::details::operator|;
    using Kokkos::Experimental::graph::parallel_for;
    using Kokkos::Experimental::graph::RangePolicy;

    /// The first workload will run on the first execution space instance execution context.
    /// Then, we transfer to the second execution space instance, and we'll need to add an implicit fence.
    auto work = this->execs.at(0)
        | parallel_for(RangePolicy(0, 1), MyDummyFunctor{.data = data})
        | continues_on(this->execs.at(1))
        | parallel_for(RangePolicy(0, 1), MyDummyFunctor{.data = data});
}


// How could we have the "single code, both modes" with multi-GPU ?
//
//    exec_0 -> on device 0
//    exec_1 -> on device 1
// 
// Graph mode:
    auto root   = create_graph()
    auto node_A = root  .then_parallel_for(Kokkos::RangePolicy(exec_0, 0, N), Functor{...}); // node placed after root,   runs on GPU 0
    auto node_B = node_A.then_parallel_for(Kokkos::RangePolicy(exec_1, 0, N), Functor{...}); // node placed after node_A, runs on GPU 1
    auto node_C = node_B.then_parallel_for(Kokkos::RangePolicy(exec_0, 0, N), Functor{...}); // node placed after node_B, runs on GPU 0

// Graph mode à la P2300
    auto root   = create_graph()
    auto node_A = root   | parallel_for(Kokkos::RangePolicy(exec_0, 0, N), Functor{...}); // node placed after root,   runs on GPU 0
    auto node_B = node_A | parallel_for(Kokkos::RangePolicy(exec_1, 0, N), Functor{...}); // node placed after node_A, runs on GPU 1
    auto node_C = node_B | parallel_for(Kokkos::RangePolicy(        0, N), Functor{...}); // node placed on device of defaulted exec DANGER
                                                                                          //  -> we need an additional state for RangePolicy (exec given or not)

// Exec space mode, keeping the same code as graph à la P2300 ? CHOICE 1
    auto root   = exec_0;
    auto node_A = root   | parallel_for(Kokkos::RangePolicy(exec_0, 0, N), Functor{...}); 
    auto node_B = node_A | parallel_for(Kokkos::RangePolicy(exec_1, 0, N), Functor{...});
    auto node_C = node_B | parallel_for(Kokkos::RangePolicy(exec_0, 0, N), Functor{...});  

// The only solution would be to always require the continues on, disallow exec space instances in the range policy, and use
// continues_on / on to place the kernel on a given device if in graph mode.

// Let's first write the exec mode in a truly P2300 compliant syntax. CHOICE 2
    auto root   = schedule(exec_0);
    auto node_A = root   |            parallel_for(RangePolicy(0, N), Functor{...});  // we want it on GPU 0 so it is fine
    auto node_B = node_A | on(exec_1, parallel_for(RangePolicy(0, N), Functor{...})); // it will run on GPU 1
    auto node_C = node_B |            parallel_for(RangePolicy(0, N), Functor{...});  // runs on GPU 0 again


-> 2 choices:

    1. keep rangepolicy as it is but each and every sender has to be constructed with some exec.
       on / continues_on are not allowed

    2. disallow exec to be given to range policy, and use p2300 on / continues_on and so on


-> let s pick choice 2 (future proof)

-> in exec mode, we really use the execs to schedule the workloads
   but in graph mode, they are just used to tell the device ID (the impl graph will decide later what to use)
   but in this way we get the single code both modes :)


} // namespace tests::kokkos_ext
