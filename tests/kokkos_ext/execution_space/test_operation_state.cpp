#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED_DEPRECATED_ATTRIBUTES
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

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
    using clsr_t = Kokkos::Experimental::details::execution_space::ParallelForClosure<
        BulkFunctor<typename OpStateTest::view_s_t>,
        Kokkos::RangePolicy<execution_space>
    >;

    using rcvr_t = tests::stdexec::SinkReceiver;

    using op_state_t =
        Kokkos::Experimental::details::execution_space::OpState<typename OpStateTest::schedule_sender_t, rcvr_t, clsr_t>;

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

    auto op_state = Kokkos::Experimental::details::execution_space::OpState{
        ::stdexec::schedule(esc.get_scheduler()), tests::stdexec::SinkReceiver{}, std::move(clsr)};

    ASSERT_EQ(Kokkos::Experimental::details::execution_space::get_exec(op_state).get(), exec);

    op_state.start();
    exec.fence();

    ASSERT_EQ(witness(), size / 2 * (size - 1) - 1);
}

} // namespace tests::kokkos_ext
