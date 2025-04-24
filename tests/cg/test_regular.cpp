#include "gtest/gtest.h"

#include "Kokkos_InnerProductSpaceTraits.hpp"
#include "Kokkos_Profiling_ProfileSection.hpp"
#include "Kokkos_Profiling_ScopedRegion.hpp"

#include "KokkosBlas1_update.hpp"
#include "KokkosSparse_spmv.hpp"

#include "kokkos-utils/callbacks/EventInProfileSectionRegexMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/concepts/ExecutionSpace.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/cg/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Conjugate gradient solver with regular @c Kokkos execution space instances
 * --------------------------------------------------------------------------
 *
 * Implement a portable conjugate gradient (CG) solver without using @c Kokkos::Graph.
 *
 * The test can be found in @ref tests/cg/test_regular.cpp.
 */

namespace tests::cg
{

//! Conjugate gradient solver with regular @c Kokkos execution space instances.
template <typename MatrixType, typename VectorType>
struct CGRegular : public ConjugateGradientSolverBase<MatrixType, VectorType>
{
    using base_t = ConjugateGradientSolverBase<MatrixType, VectorType>;

    using typename base_t::dot_t;
    using typename base_t::mag_t;

    MatrixType mat;
    VectorType rhs;

    /// We explicitly don't partition @p exec.
    /// This decision is aligned with how @c KokkosKernels would use the incoming @p exec.
    template <Kokkos::utils::concepts::ExecutionSpace Exec, typename SizeType = typename Exec::size_type>
    std::tuple<mag_t, SizeType> apply(const Exec& exec, const VectorType& sol, const mag_t tol, const SizeType max_iter) const
    {
        using spmv_handle_t = KokkosSparse::SPMVHandle<typename VectorType::memory_space, MatrixType, VectorType, VectorType>;
        spmv_handle_t handle {};

        /// As stated in https://docs.nvidia.com/cuda/cublas/#scalar-parameters, @c cuBLAS can deal with scalar parameters in 2 ways:
        ///     * @c CUBLAS_POINTER_MODE_HOST
        ///     * @c CUBLAS_POINTER_MODE_DEVICE
        /// Let's check that we are in the @c CUBLAS_POINTER_MODE_HOST mode, which implies that:
        ///     1. For methods that take scalar parameters (*e.g.* @c axpy), they shouldn’t be placed in managed memory. The kernel will use its own copy of the variables.
        ///     2. For methods that return a scalar value (*e.g. @c dot), the @c cuBLAS call will block the @c CPU thread until the kernel is finished.
        ///
        /// So using scalar values on the stack is fine, and for this implementation of the @c CG that uses a single execution space instance, it's even
        /// better than using device or host pinned views for storing intermediate variables (because it makes the whole thing more readable and not less efficient).
        ///
        /// Note that the @c CUBLAS_POINTER_MODE_DEVICE mode can be interesting. Quoting:
        ///     For example, this situation can arise when iterative methods for solution of linear systems and eigenvalue problems are implemented using the cuBLAS library.
#if defined(KOKKOS_ENABLE_CUDA)
        #if !defined(KOKKOSKERNELS_ENABLE_TPL_CUBLAS)
            #error "Kokkos Kernels TPL cuBLAS not enabled."
        #endif
        KokkosBlas::Impl::CudaBlasSingleton& s = KokkosBlas::Impl::CudaBlasSingleton::singleton();
        cublasPointerMode_t mode;
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasGetPointerMode(s.handle, &mode));
        if(mode != CUBLAS_POINTER_MODE_HOST)
            Kokkos::abort("cuBLAS pointer mode is not host.");
#endif

        //! Pre-compute the residual.
        VectorType res(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "residual"), rhs.size());
        Kokkos::deep_copy(exec, res, rhs);
        KokkosSparse::spmv(exec, &handle, "N", -1., mat, sol, 1., res);

        //! Compute the residual norm, because it might already be all good.
        dot_t res_dot_old = KokkosBlas::dot(exec, res, res);

        //! Placeholder for the residual L2 norm.
        mag_t res_nrm2 = Kokkos::sqrt(Kokkos::abs(res_dot_old));
        if(res_nrm2 < tol) return {res_nrm2, 0};

        //! Direction of search is set to the residual.
        const VectorType dir(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "dir"), rhs.size());
        Kokkos::deep_copy(exec, dir, res);

        //! Placeholder for the product of @ref mat with the direction of search.
        const VectorType mat_dir(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "mat * dir"), rhs.size());

        /// For @c Cuda, the @c dot wil not end up in a @c cuBLAS call (see https://github.com/kokkos/kokkos-kernels/blob/9bca19c85b88aeca97209ec7cde858447e16696c/blas/tpls/KokkosBlas1_dot_tpl_spec_avail.hpp#L81-L90).
        /// Therefore, @c KokkosBlas::dot will end up doing a @c Kokkos parallel reduce.
        /// If the result variable is a host scalar, it ends up making 2 fences (instead of one, see below):
        ///     - in @c Kokkos parallel reduce itself (see *e.g.* https://github.com/kokkos/kokkos/blob/c2a5c01699048e80a8ddce9d99c0050b70238b7c/core/src/Cuda/Kokkos_Cuda_Parallel_MDRange.hpp#L455-L470)
        ///     - in @c KokkosBlas (see *e.g.* https://github.com/kokkos/kokkos-kernels/blob/9bca19c85b88aeca97209ec7cde858447e16696c/blas/src/KokkosBlas1_dot.hpp#L107)
        /// To ensure we control fencing, we'll use a host pinned variable for intermediate @c dot results.
        const Kokkos::View<dot_t, Kokkos::SharedHostPinnedSpace> pinned("intermediate dot result");

        //! Loop until the norm of the residual is smaller than @p tol or the maximum number of iterations is reached.
        Kokkos::Profiling::ProfilingSection profile_section("CG iteration 7");
        SizeType iter = 0;
        while(res_nrm2 > tol && iter < max_iter)
        {
            if(iter == 7) profile_section.start();

            //! Compute @c alpha.
            KokkosSparse::spmv(exec, &handle, "N", 1., mat, dir, 0., mat_dir);

            KokkosBlas::dot(exec, pinned, dir, mat_dir);

            exec.fence("waiting for dot (quadratic)");

            const auto alpha = res_dot_old / pinned();

            /// Update the solution candidate and residual.
            /// These two updates are independant, but since we have only one execution space instance, all we gain is
            /// the launch overhead.
            KokkosBlas::axpby(exec,   alpha, dir,     1., sol);
            KokkosBlas::axpby(exec, - alpha, mat_dir, 1., res);

            //! At this point, we can already check the condition and exit.
            KokkosBlas::dot(exec, pinned, res, res);

            exec.fence("waiting for dot (norm)");

            if((res_nrm2 = Kokkos::sqrt(Kokkos::abs(pinned()))) > tol)
            {
                /// Compute @c beta.
                /// @note Though it should be purely real, floating point errors add up and it has a very tiny imaginary part.
                const auto beta = pinned() / res_dot_old;

                //! Update search direction.
                KokkosBlas::axpby(exec, 1., res, beta, dir);

                res_dot_old = pinned();
            }

            if(iter == 7) profile_section.stop();

            ++iter;
        }

        return std::tuple{res_nrm2, iter};
    }
};

using namespace Kokkos::utils::callbacks;

class CGRegularTest : public NbyNSolverTest<CGRegular<NbyNSolverTestHelper::initializer_t::matrix_t, NbyNSolverTestHelper::initializer_t::rhs_t>>,
                      public Kokkos::utils::callbacks::ManagerTestFixture
{
public:
    using event_types_list_t = Kokkos::Impl::type_list<BeginFenceEvent, BeginParallelForEvent, BeginParallelReduceEvent, PushRegionEvent, PopRegionEvent, ProfileEvent, CreateProfileSectionEvent, StartProfileSectionEvent, StopProfileSectionEvent>;

    using event_in_profile_section_recorder_t = RecorderListener<EventInProfileSectionRegexMatcher, event_types_list_t>;
};

TEST_F(CGRegularTest, 10x10)
{
    const NbyNSolverTestHelper::execution_space exec {};

#if defined(KOKKOS_ENABLE_CUDA)
    EventInProfileSectionRegexMatcher matcher(std::regex("CG iteration 7"));
    const auto recorder = std::make_shared<event_in_profile_section_recorder_t>(std::move(matcher));

    Manager::register_listener(recorder);
#endif

    this->run(exec, 10);

#if defined(KOKKOS_ENABLE_CUDA)
    Manager::unregister_listener(recorder.get());

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
