#include "kokkos_ext/impl/execution_space/operation_state.hpp"
#include "kokkos_ext/impl/execution_space/parallel_for.hpp"

#include "tests/kokkos_ext/Helpers.hpp"
#include "tests/kokkos_ext/execution_space/Helpers.hpp"
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Operation state @c Kokkos::Experimental::details::execution_space::OpState
 * --------------------------------------------------------------------------
 *
 * This group of tests check @ref Kokkos::Experimental::details::execution_space::OpState
 * and its related types.
 *
 * The tests can be found in @ref tests/kokkos_ext/execution_space/test_operation_state.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext {

class OpStateTest : public impl::ExecutionSpaceContextTest<execution_space> { };

//! @test Check traits of @c Kokkos::Experimental::details::execution_space::OpState.
consteval bool test_op_state_traits() {
    //! Schedule sender.
    using schd_sndr_t = typename OpStateTest::schedule_sender_t;

    //! Parallel for closure.
    using functor_t = BulkFunctor<typename OpStateTest::view_s_t>;
    using policy_t = Kokkos::RangePolicy<execution_space>;
    using clsr_t = Kokkos::Experimental::details::execution_space::ParallelForClosure<functor_t, policy_t>;

    //! Receiver.
    using rcvr_t = tests::stdexec::SinkReceiver;

    //! Operation state.
    using op_state_t = Kokkos::Experimental::details::execution_space::OpState<schd_sndr_t, rcvr_t, clsr_t>;

    //! Models the operation state concept.
    static_assert(::stdexec::operation_state<op_state_t>);

    //! By inheriting from @c stdexec::__immovable, it is neither moveable nor copyable.
    static_assert(!std::move_constructible<op_state_t>);
    static_assert(!std::is_move_assignable_v<op_state_t>);

    static_assert(!std::copy_constructible<op_state_t>);
    static_assert(!std::is_copy_assignable_v<op_state_t>);

    return true;
}
static_assert(test_op_state_traits());

//! @test Check construction, query for execution space instance, and start.
TEST_F(OpStateTest, construct_query_and_start) {
    constexpr size_t size = 10;

    const view_s_t witness(Kokkos::view_alloc(exec, "witness - shared space"));

    const context_t esc{exec};

    Kokkos::Experimental::ParallelForData pfor_data{
        "hello from pfor", BulkFunctor{.data = witness}, Kokkos::RangePolicy(exec, 2, size)};
    Kokkos::Experimental::details::execution_space::ParallelForClosure clsr{.data = std::move(pfor_data)};

    auto op_state = Kokkos::Experimental::details::execution_space::OpState<
        typename Kokkos::Experimental::details::execution_space::Scheduler<execution_space>::Sender,
        tests::stdexec::SinkReceiver,
        decltype(clsr)
    >{::stdexec::schedule(esc.get_scheduler()), tests::stdexec::SinkReceiver{}, std::move(clsr)};

    ASSERT_EQ(Kokkos::Experimental::details::execution_space::get_exec(op_state).get(), exec);

    op_state.start();
    exec.fence();

    ASSERT_EQ(witness(), size / 2 * (size - 1) - 1);
}

//! @test Check construction of folded operation state from two parallel for senders.
TEST_F(OpStateTest, folded_from_two) {
    constexpr size_t size = 10;

    const view_s_t witness(Kokkos::view_alloc(exec, "witness - shared space"));

    const context_t esc{exec};

    auto chain = ::stdexec::schedule(esc.get_scheduler())
               | Kokkos::Experimental::parallel_for(
                     "hello from pfor", Kokkos::RangePolicy<execution_space>(0, size), BulkFunctor{.data = witness})
               | Kokkos::Experimental::parallel_for(
                     "hello again from pfor",
                     Kokkos::RangePolicy<execution_space>(0, size),
                     BulkFunctor{.data = witness});

    auto op_state = ::stdexec::connect(std::move(chain), tests::stdexec::SinkReceiver{});

    static_assert(std::same_as<
                  decltype(op_state),
                  Kokkos::Experimental::details::execution_space::OpState<
                      Kokkos::Experimental::ParallelForSender<
                          Kokkos::Experimental::details::execution_space::Scheduler<execution_space>::Sender,
                          tests::kokkos_ext::BulkFunctor<view_s_t>,
                          Kokkos::RangePolicy<execution_space>
                      >,
                      tests::stdexec::SinkReceiver,
                      Kokkos::Experimental::details::execution_space::ParallelForClosure<
                          tests::kokkos_ext::BulkFunctor<view_s_t>,
                          Kokkos::RangePolicy<execution_space>
                      >
                  >
    >);

    static_assert(std::derived_from<
                  decltype(op_state),
                  Kokkos::Experimental::details::execution_space::OpState<
                      Kokkos::Experimental::details::execution_space::Scheduler<execution_space>::Sender,
                      tests::stdexec::SinkReceiver,
                      Kokkos::Experimental::details::execution_space::ParallelForClosure<
                          tests::kokkos_ext::BulkFunctor<view_s_t>,
                          Kokkos::RangePolicy<execution_space>
                      >,
                      Kokkos::Experimental::details::execution_space::ParallelForClosure<
                          tests::kokkos_ext::BulkFunctor<view_s_t>,
                          Kokkos::RangePolicy<execution_space>
                      >
                  >
    >);

    static_assert(::stdexec::__tuple_size_v<typename decltype(op_state)::closures_t> == 2);

    op_state.start();
    exec.fence();

    ASSERT_EQ(witness(), 2 * size / 2 * (size - 1));
}

//! @test Check construction of folded operation state from three parallel for senders.
TEST_F(OpStateTest, folded_from_three) {
    constexpr size_t size = 10;

    const view_s_t witness(Kokkos::view_alloc(exec, "witness - shared space"));

    const context_t esc{exec};

    auto chain = ::stdexec::schedule(esc.get_scheduler())
               | Kokkos::Experimental::parallel_for(
                     Kokkos::RangePolicy<execution_space>(0, size), BulkFunctor{.data = witness})
               | Kokkos::Experimental::parallel_for(
                     Kokkos::RangePolicy<execution_space>(0, size), BulkFunctor{.data = witness})
               | Kokkos::Experimental::parallel_for(
                     Kokkos::RangePolicy<execution_space>(0, size), BulkFunctor{.data = witness});

    auto op_state = ::stdexec::connect(std::move(chain), tests::stdexec::SinkReceiver{});

    static_assert(::stdexec::__tuple_size_v<typename decltype(op_state)::closures_t> == 3);

    op_state.start();
    exec.fence();

    ASSERT_EQ(witness(), 3 * size / 2 * (size - 1));
}

} // namespace tests::kokkos_ext
