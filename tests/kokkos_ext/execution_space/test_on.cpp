#include "gtest/gtest.h"

#include "kokkos-utils/callbacks/Helpers.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"

#include "tests/kokkos_ext/execution_space/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c on by @c Kokkos::Experimental::ExecutionSpaceContext
 * ------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::ExecutionSpaceContext properly customizes
 * @c on.
 *
 * The tests can be found in @ref tests/kokkos_ext/execution_space/test_on.cpp.
 */

using      execution_space = Kokkos::DefaultExecutionSpace;
using         memory_space = typename execution_space::memory_space;
using host_execution_space = Kokkos::DefaultHostExecutionSpace;

namespace tests::kokkos_ext
{

using namespace Kokkos::utils::callbacks;

class OnTest : public impl::ExecutionSpaceContextTest<execution_space, host_execution_space>,
               public Kokkos::utils::callbacks::ManagerTestFixture
{
public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent>;
    using variant_t           = std::variant    <BeginFenceEvent, BeginParallelForEvent>;
};

/**
 * @test Check that @ref Kokkos::Experimental::ExecutionSpaceContext supports @c on, using the same execution space instance.
 *
 * It should not make extra unneeded fence.
 */
TEST_F(OnTest, one_execution_space_instance)
{
    const view_t data(Kokkos::view_alloc("data", exec));

    const context_t esc{exec};

    auto chain = ::stdexec::schedule(esc.get_scheduler())
        | ADD_THEN
        | ::stdexec::v2::on(esc.get_scheduler(), ADD_THEN)
        | ADD_THEN;

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)] () mutable { ::stdexec::sync_wait(std::move(chain)); }),
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR (exec, then),
            MATCHER_FOR_BEGIN_PFOR (exec, then),
            MATCHER_FOR_BEGIN_PFOR (exec, then),
            MATCHER_FOR_BEGIN_FENCE(exec, sync_wait)
        )
    );

    const auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data);

    ASSERT_EQ(mirror(), 3);
}

/**
 * @test Check that @ref Kokkos::Experimental::ExecutionSpaceContext supports @c on, using different execution space instances
 *       of the same type.
 *
 * Proper fencing is required when transitioning from one execution space instance to another.
 */
TEST_F(OnTest, many_execution_space_instances_of_same_type)
{
    const auto execs = Kokkos::Experimental::partition_space(exec, 1, 1);

    const view_t data(Kokkos::view_alloc("data", execs.at(0)));

    const context_t esc_0{execs.at(0)}, esc_1{execs.at(1)};

    auto chain = ::stdexec::schedule(esc_0.get_scheduler())
        | ADD_THEN
        | ::stdexec::v2::on(esc_1.get_scheduler(), ADD_THEN)
        | ADD_THEN;

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)] () mutable { ::stdexec::sync_wait(std::move(chain)); }),
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR (execs.at(0), then),
            // MATCHER_FOR_BEGIN_FENCE(execs.at(0), schedule_from),
            MATCHER_FOR_BEGIN_PFOR (execs.at(1), then),
            // MATCHER_FOR_BEGIN_FENCE(execs.at(1), schedule_from),
            MATCHER_FOR_BEGIN_PFOR (execs.at(0), then),
            MATCHER_FOR_BEGIN_FENCE(execs.at(0), sync_wait)
        )
    );

    const auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data);

    ASSERT_EQ(mirror(), 3);
}

/**
 * @test Check that @ref Kokkos::Experimental::ExecutionSpaceContext supports @c on, using different execution space instances
 *       of the different types.
 *
 * Proper fencing is required when transitioning from one execution space instance to another.
 */
TEST_F(OnTest, many_execution_space_instances_of_different_type)
{
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const host_execution_space exec_h{};

    const context_h_t esc_h{exec_h};
    const context_t   esc  {exec};

    auto chain = ::stdexec::schedule(esc.get_scheduler())
        | ADD_THEN
        | ::stdexec::v2::on(esc_h.get_scheduler(), ADD_THEN)
        | ADD_THEN;

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)] () mutable { ::stdexec::sync_wait(std::move(chain)); }),
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR (exec,   then),
            // MATCHER_FOR_BEGIN_FENCE(exec,   schedule_from),
            MATCHER_FOR_BEGIN_PFOR (exec_h, then),
            // MATCHER_FOR_BEGIN_FENCE(exec_h, schedule_from),
            MATCHER_FOR_BEGIN_PFOR (exec,   then),
            MATCHER_FOR_BEGIN_FENCE(exec,   sync_wait)
        )
    );

    // ASSERT_EQ(data(), 3); reenable when fences are implemented
}

} // namespace tests::kokkos_ext
