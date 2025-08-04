#include "gtest/gtest.h"

#include "kokkos-utils/callbacks/EventNameMatcher.hpp"
#include "kokkos-utils/callbacks/EventRegionMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "algorithms/cg/Functors.hpp"
#define GRAPH_DISPATCHING_ALGORITHMS_CG_CGSINGLEQUEUE_ENABLE_SCOPEDREGION_IN_LOOP
#include "algorithms/cg/SingleQueue.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/cg/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Conjugate gradient solver with a single @c Kokkos execution space instance
 * --------------------------------------------------------------------------
 *
 * This group of tests check the behavior of @ref algorithms::cg::CGSingleQueue.
 *
 * The test can be found in @ref tests/cg/test_single_queue.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::cg
{

using helper_t = NbyNSolverTestHelper<execution_space>;

DEFINE_FUNCTOR(KokkosSparseSpmv, KokkosSparse::spmv)
DEFINE_FUNCTOR(KokkosBlasDot,    KokkosBlas::dot)
DEFINE_FUNCTOR(KokkosBlasAxpby,  KokkosBlas::axpby)

using solver_t = algorithms::cg::CGSingleQueue<
    typename helper_t::initializer_t::matrix_t,
    typename helper_t::initializer_t::rhs_t,
    KokkosSparseSpmv,
    KokkosBlasDot,
    KokkosBlasAxpby
>;

using namespace Kokkos::utils::callbacks;

class CGSingleQueueTest : public ::testing::Test,
                          public NbyNSolverTest<solver_t>,
                          public Kokkos::utils::tests::scoped::callbacks::Manager
{
public:
    using event_types_list_t = Kokkos::Impl::type_list<BeginFenceEvent, BeginParallelForEvent, BeginParallelReduceEvent, PushRegionEvent, PopRegionEvent>;

    using event_in_region_recorder_t = RecorderListener<EventRegionMatcher<EventNameMatcher>, event_types_list_t>;
};

TEST_F(CGSingleQueueTest, 10x10)
{
    const execution_space exec {};

#if defined(KOKKOS_ENABLE_CUDA)
    EventRegionMatcher matcher{.matcher = EventNameMatcher{.name = "CGSingleQueue - iter 7"}};
    const auto recorder = std::make_shared<event_in_region_recorder_t>(std::move(matcher));

    Kokkos::utils::callbacks::Manager::register_listener(recorder);
#endif

    RUN_AND_CHECK(exec, 10, 1.e-12, 9)

#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::utils::callbacks::Manager::unregister_listener(recorder.get());

    /// When the backend is @c Cuda, there won't be any call to @c cublasDdot because of
    /// https://github.com/kokkos/kokkos-kernels/blob/9bca19c85b88aeca97209ec7cde858447e16696c/blas/tpls/KokkosBlas1_dot_tpl_spec_avail.hpp#L81-L90.
    ASSERT_THAT(
        recorder->recorded_events,
        ::testing::ElementsAre(
            MATCHER_FOR_PUSH_REGION("KokkosSparse::spmv[TPL_CUSPARSE,Kokkos::complex<double>]"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::dot[noETI]"),
                MATCHER_FOR_BEGIN_PRED(exec, "KokkosBlas::dot<1D>"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_BEGIN_FENCE(exec, "waiting for dot (quadratic)"),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::axpby[TPL_CUBLAS,complex<double>]"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::axpby[TPL_CUBLAS,complex<double>]"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::dot[noETI]"),
                MATCHER_FOR_BEGIN_PRED(exec, "KokkosBlas::dot<1D>"),
            MATCHER_FOR_POP_REGION(),

            MATCHER_FOR_BEGIN_FENCE(exec, "waiting for dot (norm)"),

            MATCHER_FOR_PUSH_REGION("KokkosBlas::axpby[TPL_CUBLAS,complex<double>]"),
                /// There is no official support for @c axpby in @c cuBLAS. Therefore, if @c beta is not one,
                /// @c KokkosKernels will fallback to its own implementation.
                /// See also https://github.com/kokkos/kokkos-kernels/blob/9bca19c85b88aeca97209ec7cde858447e16696c/blas/tpls/KokkosBlas1_axpby_tpl_spec_decl.hpp#L289-L305.
                MATCHER_FOR_PUSH_REGION("KokkosBlas::axpby[noETI]"),
                    MATCHER_FOR_BEGIN_PFOR(exec, "KokkosBlas::Axpby::S11"),
                MATCHER_FOR_POP_REGION(),
            MATCHER_FOR_POP_REGION()
        )
    );
#endif
}

} // namespace tests::cg
