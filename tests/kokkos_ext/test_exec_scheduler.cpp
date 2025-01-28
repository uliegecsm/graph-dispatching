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

class ExecutionSpaceContextTest : public ::testing::Test
{
public:
    using view_t = Kokkos::View<int[1], Kokkos::SharedSpace>;

public:
    void SetUp() override
    {
        exec = Kokkos::Experimental::partition_space(execution_space{}, 1)[0];
        data = view_t(Kokkos::view_alloc("data", exec));
    }

protected:
    execution_space exec;
    view_t data;
};

//! @test Check that handliong of @c then.
TEST_F(ExecutionSpaceContextTest, then)
{
    //! Build the chain.
    Kokkos::Experimental::ExecutionSpaceContext esc{exec};

    auto sch = esc.get_scheduler();

    auto start = Kokkos::Experimental::schedule(std::move(sch));

    auto partial_pfor_1 = Kokkos::Experimental::graph::parallel_for(Kokkos::RangePolicy(0, 1), MyDummyFunctor{.data = data});
    auto partial_pfor_2 = Kokkos::Experimental::graph::parallel_for(Kokkos::RangePolicy(0, 1), MyDummyFunctor{.data = data});

    auto chain_1 = std::move(start)   | std::move(partial_pfor_1);
    auto chain_2 = std::move(chain_1) | std::move(partial_pfor_2);

    //! Check types.
    using sch_t = Kokkos::Experimental::details::execution_space::ExecutionSpaceScheduler<Kokkos::Cuda>;
    static_assert(std::same_as<decltype(sch), sch_t>);

    using sch_sender_t = typename sch_t::Sender;
    static_assert(std::same_as<decltype(start), sch_sender_t>);

    using partial_pfor_t = Kokkos::Experimental::graph::details::PartialAlgorithm<
        Kokkos::ParallelForTag, std::string, Kokkos::RangePolicy<>, MyDummyFunctor<view_t>
    >;
    static_assert(std::same_as<decltype(partial_pfor_1), partial_pfor_t>);
    static_assert(std::same_as<decltype(partial_pfor_2), partial_pfor_t>);

    using pfor_1_sender_t = Kokkos::Experimental::graph::details::ParallelForSender<sch_sender_t, Kokkos::RangePolicy<>, MyDummyFunctor<view_t>>;
    using pfor_2_sender_t = Kokkos::Experimental::graph::details::ParallelForSender<pfor_1_sender_t, Kokkos::RangePolicy<>, MyDummyFunctor<view_t>>;

    static_assert(std::same_as<decltype(chain_1), pfor_1_sender_t>);
    static_assert(std::same_as<decltype(chain_2), pfor_2_sender_t>);

    //! Wait for completion.
    Kokkos::Experimental::sync_wait(std::move(chain_2));

    ASSERT_EQ(data(0), 2);
}

/**
 * @test Check handling of @c when_all.
 *
 * @note Though we split in 2 possibly asynchronous branches, our @ref ExecutionSpaceContext
 *       has a single resource to schedule both workloads. They are therefore serialized.
 */
TEST_F(ExecutionSpaceContextTest, when_all)
{
    //! Build the chain.
    Kokkos::Experimental::ExecutionSpaceContext esc{exec};

    auto sch = esc.get_scheduler();

    auto start = Kokkos::Experimental::schedule(sch);

    auto partial_pfor_1 = Kokkos::Experimental::graph::parallel_for(Kokkos::RangePolicy(0, 1), MyDummyFunctor{.data = data});

    auto pfor = std::move(start) | std::move(partial_pfor_1);

    auto split = std::move(pfor) | Kokkos::Experimental::graph::split();

    auto partial_pfor_2 = Kokkos::Experimental::graph::parallel_for(Kokkos::RangePolicy(0, 1), MyDummyFunctor{.data = data});
    auto partial_pfor_3 = Kokkos::Experimental::graph::parallel_for(Kokkos::RangePolicy(0, 1), MyDummyFunctor{.data = data});

    auto branch_1 =           split  | std::move(partial_pfor_2);
    auto branch_2 = std::move(split) | std::move(partial_pfor_3);

    auto join = Kokkos::Experimental::graph::when_all(std::move(branch_1), std::move(branch_2));

    auto chain = std::move(join) | Kokkos::Experimental::graph::continues_on(sch) | std::move(partial_pfor_3);

    //! Check types.
    using sch_t = Kokkos::Experimental::details::execution_space::ExecutionSpaceScheduler<Kokkos::Cuda>;
    static_assert(std::same_as<decltype(sch), sch_t>);

    using sch_sender_t = typename sch_t::Sender;
    static_assert(std::same_as<decltype(start), sch_sender_t>);

    using pfor_t = Kokkos::Experimental::graph::details::ParallelForSender<sch_sender_t, Kokkos::RangePolicy<>, MyDummyFunctor<view_t>>;
    static_assert(std::same_as<decltype(pfor), pfor_t>);

    using split_sender_t = Kokkos::Experimental::graph::details::SplitSender<pfor_t>;
    static_assert(std::same_as<decltype(split), split_sender_t>);

    using partial_pfor_t = Kokkos::Experimental::graph::details::PartialAlgorithm<
        Kokkos::ParallelForTag, std::string, Kokkos::RangePolicy<>, MyDummyFunctor<view_t>
    >;
    static_assert(std::same_as<decltype(partial_pfor_1), partial_pfor_t>);
    static_assert(std::same_as<decltype(partial_pfor_2), partial_pfor_t>);
    static_assert(std::same_as<decltype(partial_pfor_3), partial_pfor_t>);

    using pfor_sender_t = Kokkos::Experimental::graph::details::ParallelForSender<split_sender_t, Kokkos::RangePolicy<>, MyDummyFunctor<view_t>>;

    static_assert(std::same_as<decltype(branch_1), pfor_sender_t>);
    static_assert(std::same_as<decltype(branch_2), pfor_sender_t>);

    using when_all_t = Kokkos::Experimental::graph::details::WhenAllSender<
        pfor_sender_t,
        pfor_sender_t
    >;
    static_assert(std::same_as<decltype(join), when_all_t>);

    using chain_t = Kokkos::Experimental::graph::details::ParallelForSender<
        Kokkos::Experimental::graph::details::ContinuesOnSender<when_all_t, sch_t>,
        Kokkos::RangePolicy<>, MyDummyFunctor<view_t>
    >;
    static_assert(std::same_as<decltype(chain), chain_t>);

    //! Wait for completion.
    Kokkos::Experimental::sync_wait(std::move(chain));

    ASSERT_EQ(data(0), 4);
}

} // namespace tests::kokkos_ext
