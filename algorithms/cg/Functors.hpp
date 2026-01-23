#ifndef GRAPH_DISPATCHING_ALGORITHMS_CG_FUNCTORS_HPP
#define GRAPH_DISPATCHING_ALGORITHMS_CG_FUNCTORS_HPP

#include "Kokkos_Graph.hpp"

#include "KokkosSparse_spmv.hpp"

#include "algorithms/cg/Helpers.hpp"

namespace algorithms::cg
{

template <typename AType, typename BType>
struct DivideAndSwap
{
    AType a;
    BType b;

    KOKKOS_FUNCTION
    void operator()() const {
        b() = std::exchange(a(), a() / b());
    }
};

/**
 * @brief Perform a sparse matrix-vector multiply, graph-compatible.
 *
 * The implementation relies on the @c KokkosSparse kernel, inspired by:
 *  - https://github.com/trilinos/Trilinos/blob/62023ad68e09a2972240971c40be34465010d6f3/packages/kokkos-kernels/perf_test/sparse/KokkosSparse_spmv_struct_tuning.cpp#L191.
 *
 * @note Therefore, this version will never rely on a TPL. This is because @c KokkosKernels is not yet graph-compatible;
 *       calling @c KokkosSparse::spmv using a capture stream might not work either - but could be tried.
 *
 * @warning Not tuned to be efficient.
 */
template <typename Pred, typename Handle, typename Alpha, typename AMatrix, typename XVector, typename Beta, typename YVector, int DoBeta = 0, bool Conjugate = false>
decltype(auto) spmv(const Pred& pred, const Handle&, const char mode[], Alpha&& alpha, AMatrix&& mat, XVector&& vec_x, Beta&& beta, YVector&& vec_y)
{
    using execution_space = std::conditional_t<
        Kokkos::ExecutionSpace<Pred>,
        Pred,
        typename Pred::execution_space
    >;

    if(mode[0] != 'N') Kokkos::abort("Unsupported mode.");

    if constexpr (DoBeta == 0) {
        if (beta != 0) Kokkos::abort("When DoBeta==0, beta must be 0.");
    }

    const auto num_rows = mat.numRows();

    KokkosSparse::Impl::SPMV_Functor< // NOLINT(misc-const-correctness)
        execution_space,
        std::remove_cvref_t<AMatrix>,
        std::remove_cvref_t<XVector>,
        std::remove_cvref_t<YVector>,
        DoBeta,
        Conjugate
    > functor(
        std::forward<Alpha>(alpha),
        std::forward<AMatrix>(mat),
        std::forward<XVector>(vec_x),
        std::forward<Beta>(beta),
        std::forward<YVector>(vec_y),
        0
    );

    if constexpr (Kokkos::ExecutionSpace<Pred>) {
        Kokkos::parallel_for(
            "algorithms::cg::spmv",
            Kokkos::RangePolicy(pred, 0, num_rows),
            std::move(functor)
        );
    } else {
        return pred.then_parallel_for(
            "algorithms::cg::spmv",
            make_range_policy_with_graph_exec(pred, 0, num_rows),
            std::move(functor)
        );
    }
}

//! Dot product, graph-compatible.
template <typename Pred, typename Result, typename ViewX, typename ViewY>
decltype(auto) dot(const Pred& pred, Result&& result, ViewX&& vec_x, ViewY&& vec_y)
{
    using execution_space = std::conditional_t<
        Kokkos::ExecutionSpace<Pred>,
        Pred,
        typename Pred::execution_space
    >;

    KokkosBlas::Impl::DotFunctor< // NOLINT(misc-const-correctness)
        std::remove_cvref_t<Result>,
        std::remove_cvref_t<ViewX>,
        std::remove_cvref_t<ViewY>,
        typename execution_space::size_type
    > functor(std::forward<ViewX>(vec_x), std::forward<ViewY>(vec_y));

    if constexpr (Kokkos::ExecutionSpace<Pred>) {
        Kokkos::parallel_reduce(
            "algorithms::cg::dot",
            Kokkos::RangePolicy(pred, 0, functor.m_x.size()),
            std::move(functor),
            std::forward<Result>(result)
        );
    } else {
        return pred.then_parallel_reduce(
            "algorithms::cg::dot",
            make_range_policy_with_graph_exec(pred, 0, functor.m_x.size()),
            std::move(functor),
            std::forward<Result>(result)
        );
    }
}

template <typename T> requires (!Kokkos::is_view_v<std::remove_cvref_t<T>>)
constexpr decltype(auto) get_value(T&& value) {
    return std::forward<T>(value);
}

template <typename T> requires (Kokkos::is_view_v<std::remove_cvref_t<T>> && std::remove_cvref_t<T>::rank() == 0)
constexpr decltype(auto) get_value(T&& value) {
    return std::forward<T>(value)();
}

namespace impl
{
template <typename Alpha, typename ViewX, typename Beta, typename ViewY>
struct Axpby
{
    Alpha alpha;
    ViewX vec_x;
    Beta  beta;
    ViewY vec_y;

    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T index) const {
        vec_y(index) = get_value(alpha) * vec_x(index) + get_value(beta) * vec_y(index);
    }
};
} // namespace impl

/**
 * @brief Equivalent to @c KokkosBlas::axpby, graph-compatible.
 *
 * @note @c alpha and @c beta can be rank-0 views (for coefficients that vary across multiple re-submissions).
 */
template <typename Pred, typename Alpha, typename ViewX, typename Beta, typename ViewY>
decltype(auto) axpby(const Pred& pred, Alpha&& alpha, ViewX&& vec_x, Beta&& beta, ViewY&& vec_y)
{
    const auto size = vec_x.size();

    impl::Axpby< // NOLINT(misc-const-correctness)
        std::remove_cvref_t<Alpha>,
        std::remove_cvref_t<ViewX>,
        std::remove_cvref_t<Beta>,
        std::remove_cvref_t<ViewY>
    > functor{ // NOLINT(misc-const-correctness)
        .alpha = std::forward<Alpha>(alpha),
        .vec_x = std::forward<ViewX>(vec_x),
        .beta  = std::forward<Beta>(beta),
        .vec_y = std::forward<ViewY>(vec_y)
    };

    if constexpr (Kokkos::ExecutionSpace<Pred>) {
        Kokkos::parallel_for(
            "algorithms::cg::axpby",
            Kokkos::RangePolicy(pred, 0, size),
            std::move(functor)
        );
    } else {
        return pred.then_parallel_for(
            "algorithms::cg::axpby",
            make_range_policy_with_graph_exec(pred, 0, size),
            std::move(functor)
        );
    }
}

//! Helper to defined a new type whose call operator wraps another callable. // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define DEFINE_FUNCTOR(_for_, _with_)                     \
    struct _for_                                          \
    {                                                     \
        template <typename... Args>                       \
        decltype(auto) operator()(Args&&... args) const { \
            return _with_(std::forward<Args>(args)...);   \
        }                                                 \
    };

DEFINE_FUNCTOR(Spmv,  ::algorithms::cg::spmv)
DEFINE_FUNCTOR(Dot,   ::algorithms::cg::dot)
DEFINE_FUNCTOR(Axpby, ::algorithms::cg::axpby)

} // namespace algorithms::cg

#endif // GRAPH_DISPATCHING_ALGORITHMS_CG_FUNCTORS_HPP
