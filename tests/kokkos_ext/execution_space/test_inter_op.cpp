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

template <typename ValueType, bool OnDevice>
struct LoadCheckAddFunctor
{
    ValueType prev;
    ValueType value;
    ValueType* data;

    KOKKOS_FUNCTION
    void operator()() const
    {
        std::flush(std::cout);
        std::cout << "Running LoadCheckAddFunctor with prev " << prev << " and " << value << " at " << data << std::endl;
        std::flush(std::cout);
        // if constexpr (OnDevice) { KOKKOS_IF_ON_HOST  (/*Kokkos::abort*/printf("Bulk: you should not be running on host.");) }
        // else                    { KOKKOS_IF_ON_DEVICE(/*Kokkos::abort*/printf("Bulk: you should not be running on device.");) }

        // if(*data != prev) Kokkos::abort("Unexpected value.");
        *data += value;


    }
};

//! @test Check that @ref Kokkos::Experimental::ExecutionSpaceContext can be used along with @c exec::static_thread_pool.
TEST_F(InterOpTest, static_thread_pool)
{
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    using value_t = typename view_s_t::value_type;

    const Kokkos::DefaultHostExecutionSpace exec_h{};

    const context_t esc{exec};
    const context_h_t esc_h {exec_h};

    exec::static_thread_pool pool{1};

    ::stdexec::scheduler auto scheduler_exec = esc .get_scheduler();
    ::stdexec::scheduler auto scheduler_exec_h = esc_h .get_scheduler();
    ::stdexec::scheduler auto scheduler_pool = pool.get_scheduler();

    std::cout << "> exec  : " << Kokkos::Tools::Experimental::device_id(exec) << std::endl;
    std::cout << "> exec_h: " << Kokkos::Tools::Experimental::device_id(exec_h) << std::endl;

    auto chain = ::stdexec::schedule(scheduler_exec)
        | ::stdexec::then(LoadCheckAddFunctor<value_t, true>{.prev =  0, .value = 4, .data = data.data()})
        | ::stdexec::continues_on(scheduler_pool)
        | ::stdexec::then(LoadCheckAddFunctor<value_t, false>{.prev =  4, .value = 4, .data = data.data()})
        | ::stdexec::continues_on(scheduler_exec_h)
        | ::stdexec::then(LoadCheckAddFunctor<value_t, true>{.prev =  8, .value = 4, .data = data.data()});

    const auto recorded_events = recorder_listener_t::record([chain = std::move(chain)] () mutable { ::stdexec::sync_wait(std::move(chain)); });
    for (const auto& recorded_event : recorded_events) {
        std::visit([] (const auto& arg) { std::cout << "- " << arg << std::endl; }, recorded_event);
    }

    EXPECT_THAT(
        recorded_events,
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR (exec, then),
            MATCHER_FOR_BEGIN_FENCE(exec, schedule_from),
            // then on pool lol
            MATCHER_FOR_BEGIN_PFOR (exec_h, then),
            MATCHER_FOR_BEGIN_FENCE(exec_h, sync_wait)
        )
    );

    ASSERT_EQ(data(), 12);

    // run loop issue stuff if we had a transition after the thread pool.
    // must show the run_loop in the env of the sync wait receiver
}

} // namespace tests::kokkos_ext

