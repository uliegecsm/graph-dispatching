#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

#include "kokkos-utils/callbacks/EventNameMatcher.hpp"
#include "kokkos-utils/callbacks/EventRegionMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/concepts/View.hpp"
#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "algorithms/cg/Functors.hpp"
#include "algorithms/pcg/Preconditioners.hpp"
#define GRAPH_DISPATCHING_ALGORITHMS_PCG_PCGQUEUE_ENABLE_SCOPEDREGION_IN_LOOP
#include "algorithms/pcg/Queue.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/cg/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Preconditioned conjugate gradient solver with @c Kokkos execution space instances
 * ---------------------------------------------------------------------------------
 *
 * This group of tests check the behavior of @ref algorithms::pcg::PCGQueue.
 *
 * The test can be found in @ref tests/pcg/test_queue.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::pcg
{

DEFINE_FUNCTOR(KokkosSparseSpmv, KokkosSparse::spmv)
DEFINE_FUNCTOR(KokkosBlasDot,    KokkosBlas::dot)
DEFINE_FUNCTOR(KokkosBlasAxpby,  KokkosBlas::axpby)

using namespace Kokkos::utils::callbacks;

template <template <typename> class Preconditioner>
class PCGQueueTest : public ::testing::Test,
                           public Kokkos::utils::tests::scoped::callbacks::Manager,
                           public utils::ExecutionSpacePoolFixture<execution_space, 2>
{
public:
    using helper_t = cg::NbyNSolverTestHelper<execution_space>;

    using scalar_t = helper_t::initializer_t::rhs_t::non_const_value_type;

    using event_types_list_t = Kokkos::Impl::type_list<BeginFenceEvent, BeginParallelForEvent, BeginParallelReduceEvent, BeginDeepCopyEvent,  PushRegionEvent, PopRegionEvent>;

    using event_in_region_recorder_t = RecorderListener<EventRegionMatcher<EventNameMatcher>, event_types_list_t>;

    using matrix_t = helper_t::initializer_t::matrix_t;
    using rhs_t    = helper_t::initializer_t::rhs_t;

    using preconditioner_t = Preconditioner<matrix_t>;

    using solver_t = ::algorithms::pcg::PCGQueue<
        matrix_t,
        rhs_t,
        preconditioner_t,
        KokkosSparseSpmv,
        KokkosBlasDot,
        KokkosBlasAxpby
    >;
};

#define TEST_BASE(_preconditioner_)                                       \
    class PCGQueue##_preconditioner_##Test                          \
        : public PCGQueueTest<::algorithms::pcg::_preconditioner_>, \
          public cg::NbyNSolverTest<typename PCGQueueTest<::algorithms::pcg::_preconditioner_>::solver_t> {};

TEST_BASE(IdentityPreconditioner)
TEST_BASE(DiagonalPreconditioner)
TEST_BASE(JacobiPreconditioner)

/**
 * @test The PCG must behave like the CG when the preconditioner is the identity.
 *
 * It uses the @ref algorithms::pcg::IdentityPreconditioner preconditioner.
 *
 * This is an important sanity check.
 */
TEST_F(PCGQueueIdentityPreconditionerTest, 10x10)
{
    static_assert(Kokkos::utils::impl::is_instance_of_v<preconditioner_t, ::algorithms::pcg::IdentityPreconditioner>);

#if defined(KOKKOS_ENABLE_CUDA)
    EventRegionMatcher matcher{.matcher = EventNameMatcher{.name = "PCGQueue - iter 7"}};
    const auto recorder = std::make_shared<event_in_region_recorder_t>(std::move(matcher));

    Kokkos::utils::callbacks::Manager::register_listener(recorder);
#endif

    RUN_AND_CHECK(this->pool, 10, 1.e-12, 9)

#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::utils::callbacks::Manager::unregister_listener(recorder.get());

    recorder->report(std::cout);

    const auto& exec_A = pool.get(0);
    const auto& exec_B = pool.get(1);

    /// When the backend is @c Cuda, there won't be any call to @c cublasDdot because of
    /// https://github.com/kokkos/kokkos-kernels/blob/9bca19c85b88aeca97209ec7cde858447e16696c/blas/tpls/KokkosBlas1_dot_tpl_spec_avail.hpp#L81-L90.
    ASSERT_THAT(
        recorder->recorded_events,
        ::testing::ElementsAre(
            MATCHER_FOR_PUSH_REGION("KokkosSparse::spmv[TPL_CUSPARSE,Kokkos::complex<double>]"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::dot[noETI]"),
                MATCHER_FOR_BEGIN_PRED(exec_A, "KokkosBlas::dot<1D>"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_BEGIN_FENCE(exec_A, "waiting for dot (quadratic)"),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::axpby[TPL_CUBLAS,complex<double>]"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::axpby[TPL_CUBLAS,complex<double>]"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::dot[noETI]"),
                MATCHER_FOR_BEGIN_PRED(exec_B, "KokkosBlas::dot<1D>"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_BEGIN_FENCE(exec_B, "waiting for dot (norm)"),

            MATCHER_FOR_BEGIN_DEEP_COPY(
                (AllocDescriptor{.kpsh = {.name = "Cuda"}, .name = "residual of preconditioned system", .size = 10 * sizeof(scalar_t)}),
                (AllocDescriptor{.kpsh = {.name = "Cuda"}, .name = "residual",                          .size = 10 * sizeof(scalar_t)})
            ),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::dot[noETI]"),
                MATCHER_FOR_BEGIN_PRED(exec_B, "KokkosBlas::dot<1D>"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_BEGIN_FENCE(exec_B, "waiting for dot (residuals)"),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::axpby[TPL_CUBLAS,complex<double>]"),
                MATCHER_FOR_PUSH_REGION("KokkosBlas::axpby[noETI]"),
                    MATCHER_FOR_BEGIN_PFOR(exec_A, "KokkosBlas::Axpby::S11"),
                MATCHER_FOR_POP_REGION(),
            MATCHER_FOR_POP_REGION()
        )
    );
#endif
}

//! @test Use @ref algorithms::pcg::DiagonalPreconditioner as preconditioner.
TEST_F(PCGQueueDiagonalPreconditionerTest, 10x10)
{
    static_assert(Kokkos::utils::impl::is_instance_of_v<preconditioner_t, ::algorithms::pcg::DiagonalPreconditioner>);

#if defined(KOKKOS_ENABLE_CUDA)
    EventRegionMatcher matcher{.matcher = EventNameMatcher{.name = "PCGQueue - iter 6"}};
    const auto recorder = std::make_shared<event_in_region_recorder_t>(std::move(matcher));

    Kokkos::utils::callbacks::Manager::register_listener(recorder);
#endif

    RUN_AND_CHECK(this->pool, 10, 1.e-12, 8)

#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::utils::callbacks::Manager::unregister_listener(recorder.get());

    recorder->report(std::cout);

    const auto& exec_A = pool.get(0);
    const auto& exec_B = pool.get(1);

    /// When the backend is @c Cuda, there won't be any call to @c cublasDdot because of
    /// https://github.com/kokkos/kokkos-kernels/blob/9bca19c85b88aeca97209ec7cde858447e16696c/blas/tpls/KokkosBlas1_dot_tpl_spec_avail.hpp#L81-L90.
    ASSERT_THAT(
        recorder->recorded_events,
        ::testing::ElementsAre(
            MATCHER_FOR_PUSH_REGION("KokkosSparse::spmv[TPL_CUSPARSE,Kokkos::complex<double>]"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::dot[noETI]"),
                MATCHER_FOR_BEGIN_PRED(exec_A, "KokkosBlas::dot<1D>"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_BEGIN_FENCE(exec_A, "waiting for dot (quadratic)"),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::axpby[TPL_CUBLAS,complex<double>]"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::axpby[TPL_CUBLAS,complex<double>]"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::dot[noETI]"),
                MATCHER_FOR_BEGIN_PRED(exec_B, "KokkosBlas::dot<1D>"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_BEGIN_FENCE(exec_B, "waiting for dot (norm)"),

            MATCHER_FOR_BEGIN_PFOR(exec_B, (Kokkos::Impl::TypeInfo<typename preconditioner_t::Apply<rhs_t, rhs_t>>::name())),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::dot[noETI]"),
                MATCHER_FOR_BEGIN_PRED(exec_B, "KokkosBlas::dot<1D>"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_BEGIN_FENCE(exec_B, "waiting for dot (residuals)"),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::axpby[TPL_CUBLAS,complex<double>]"),
                MATCHER_FOR_PUSH_REGION("KokkosBlas::axpby[noETI]"),
                    MATCHER_FOR_BEGIN_PFOR(exec_A, "KokkosBlas::Axpby::S11"),
                MATCHER_FOR_POP_REGION(),
            MATCHER_FOR_POP_REGION()
        )
    );
#endif
}

//! @test Use @ref algorithms::pcg::JacobiPreconditioner as preconditioner.
TEST_F(PCGQueueJacobiPreconditionerTest, 10x10)
{
    static_assert(Kokkos::utils::impl::is_instance_of_v<preconditioner_t, ::algorithms::pcg::JacobiPreconditioner>);

#if defined(KOKKOS_ENABLE_CUDA)
    EventRegionMatcher matcher{.matcher = EventNameMatcher{.name = "PCGQueue - iter 2"}};
    const auto recorder = std::make_shared<event_in_region_recorder_t>(std::move(matcher));

    Kokkos::utils::callbacks::Manager::register_listener(recorder);
#endif

    RUN_AND_CHECK(this->pool, 10, 1.e-12, 4)

#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::utils::callbacks::Manager::unregister_listener(recorder.get());

    recorder->report(std::cout);

    const auto& exec_A = pool.get(0);
    const auto& exec_B = pool.get(1);

    /// When the backend is @c Cuda, there won't be any call to @c cublasDdot because of
    /// https://github.com/kokkos/kokkos-kernels/blob/9bca19c85b88aeca97209ec7cde858447e16696c/blas/tpls/KokkosBlas1_dot_tpl_spec_avail.hpp#L81-L90.
    ASSERT_THAT(
        recorder->recorded_events,
        ::testing::ElementsAre(
            MATCHER_FOR_PUSH_REGION("KokkosSparse::spmv[TPL_CUSPARSE,Kokkos::complex<double>]"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::dot[noETI]"),
                MATCHER_FOR_BEGIN_PRED(exec_A, "KokkosBlas::dot<1D>"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_BEGIN_FENCE(exec_A, "waiting for dot (quadratic)"),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::axpby[TPL_CUBLAS,complex<double>]"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::axpby[TPL_CUBLAS,complex<double>]"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::dot[noETI]"),
                MATCHER_FOR_BEGIN_PRED(exec_B, "KokkosBlas::dot<1D>"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_BEGIN_FENCE(exec_B, "waiting for dot (norm)"),

            MATCHER_FOR_BEGIN_PFOR(exec_B, (Kokkos::Impl::TypeInfo<typename preconditioner_t::ApplyFirstPass<rhs_t, rhs_t>>::name())),
            MATCHER_FOR_BEGIN_PFOR(exec_B, (Kokkos::Impl::TypeInfo<typename preconditioner_t::Apply<rhs_t, rhs_t>>::name())),
            MATCHER_FOR_BEGIN_DEEP_COPY(
                (AllocDescriptor{.kpsh = {.name = "Cuda"}, .name = "residual of preconditioned system", .size = 10 * sizeof(scalar_t)}),
                (AllocDescriptor{.kpsh = {.name = "Cuda"}, .name = "Jacobi temporary storage",          .size = 10 * sizeof(scalar_t)})
            ),
            MATCHER_FOR_BEGIN_PFOR(exec_B, (Kokkos::Impl::TypeInfo<typename preconditioner_t::Apply<rhs_t, rhs_t>>::name())),
            MATCHER_FOR_BEGIN_DEEP_COPY(
                (AllocDescriptor{.kpsh = {.name = "Cuda"}, .name = "residual of preconditioned system", .size = 10 * sizeof(scalar_t)}),
                (AllocDescriptor{.kpsh = {.name = "Cuda"}, .name = "Jacobi temporary storage",          .size = 10 * sizeof(scalar_t)})
            ),
            MATCHER_FOR_BEGIN_PFOR(exec_B, (Kokkos::Impl::TypeInfo<typename preconditioner_t::Apply<rhs_t, rhs_t>>::name())),
            MATCHER_FOR_BEGIN_DEEP_COPY(
                (AllocDescriptor{.kpsh = {.name = "Cuda"}, .name = "residual of preconditioned system", .size = 10 * sizeof(scalar_t)}),
                (AllocDescriptor{.kpsh = {.name = "Cuda"}, .name = "Jacobi temporary storage",          .size = 10 * sizeof(scalar_t)})
            ),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::dot[noETI]"),
                MATCHER_FOR_BEGIN_PRED(exec_B, "KokkosBlas::dot<1D>"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_BEGIN_FENCE(exec_B, "waiting for dot (residuals)"),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::axpby[TPL_CUBLAS,complex<double>]"),
                MATCHER_FOR_PUSH_REGION("KokkosBlas::axpby[noETI]"),
                    MATCHER_FOR_BEGIN_PFOR(exec_A, "KokkosBlas::Axpby::S11"),
                MATCHER_FOR_POP_REGION(),
            MATCHER_FOR_POP_REGION()
        )
    );
#endif
}

} // namespace tests::pcg
