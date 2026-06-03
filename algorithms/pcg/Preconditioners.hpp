#ifndef GRAPH_DISPATCHING_ALGORITHMS_PCG_PRECONDITIONERS_HPP
#define GRAPH_DISPATCHING_ALGORITHMS_PCG_PRECONDITIONERS_HPP

#include <concepts>

#include "Kokkos_Graph.hpp"

#include "kokkos-utils/concepts/View.hpp"
#include "kokkos-utils/impl/type_traits.hpp"

#include "algorithms/cg/Helpers.hpp"

template <typename T>
concept IsGraphNode = Kokkos::utils::impl::InstanceOf<T, Kokkos::Experimental::GraphNodeRef>;

namespace algorithms::pcg
{
template <typename, typename>
struct DeepCopy;

//! Specialization for copying a rank-1 view into another rank-1 view.
template <Kokkos::utils::concepts::ViewOfRank<1> DstType, Kokkos::utils::concepts::ViewOfRank<1> SrcType>
struct DeepCopy<DstType, SrcType>
{
    typename DstType::non_const_type dst;
    typename SrcType::const_type src;

    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T index) const {
        dst(index) = src(index);
    }
};

//! Specialization for copying a scalar into a rank-1 view.
template <Kokkos::utils::concepts::ViewOfRank<1> DstType, typename SrcType> requires std::is_scalar_v<SrcType>
struct DeepCopy<DstType, SrcType>
{
    typename DstType::non_const_type dst;
    SrcType src;

    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T index) const {
        dst(index) = src;
    }
};

//! @todo Use a proper @c memcpy node.
template <typename Pred, typename DstType, typename SrcType> requires (
    Kokkos::utils::concepts::ViewOfRank<std::remove_cvref_t<DstType>, 1>
)
decltype(auto) then_deep_copy(const Pred& pred, DstType&& dst, SrcType&& src)
{
    using execution_space = typename std::remove_cvref_t<Pred>::execution_space;
    using deep_copy_t = DeepCopy<std::remove_cvref_t<DstType>, std::remove_cvref_t<SrcType>>;

    const auto size = dst.size();

    return pred.then_parallel_for(
        Kokkos::Experimental::node_props("deep copy"),
        Kokkos::RangePolicy<execution_space>(0, size),
        deep_copy_t{.dst = std::forward<DstType>(dst), .src = std::forward<SrcType>(src)}
    );
}

//! Identity.
template <typename MatrixType>
struct IdentityPreconditioner
{
    template <Kokkos::ExecutionSpace Exec>
    IdentityPreconditioner(const Exec&, const MatrixType&) {}

    //! @note It must be a deep copy because it's not supposed to touch the address of @p dst.
    template <Kokkos::ExecutionSpace Exec, Kokkos::utils::concepts::ViewOfRank<1> VectorType>
    void apply(const Exec& exec, const VectorType& dst, const VectorType& src) const {
        Kokkos::deep_copy(exec, dst, src);
    }

    template <typename Pred, typename DstType, typename SrcType> requires IsGraphNode<std::remove_cvref_t<Pred>>
    decltype(auto) apply(const Pred& pred, DstType&& dst, SrcType&& src) const {
        return then_deep_copy(pred, std::forward<DstType>(dst), std::forward<SrcType>(src));
    }
};

//! Diagonal scaling. Simple, but not very efficient in general.
template <typename MatrixType>
struct DiagonalPreconditioner
{
    using       values_view_t = typename MatrixType::values_type::non_const_type;
    using const_values_view_t = typename values_view_t::const_type;

    //! The constructor will pre-compute the inverse of the diagonal values.
    const_values_view_t values;

    template <Kokkos::ExecutionSpace Exec>
    DiagonalPreconditioner(const Exec& exec, const typename MatrixType::const_type& mat) {
        this->init(exec, mat);
    }

    template <typename Exec>
    void init(const Exec& exec, const typename MatrixType::const_type& mat)
    {
        values_view_t tmp(Kokkos::view_alloc(Kokkos::WithoutInitializing, "inverse of diagonal values", exec), mat.numRows()); // NOLINT(misc-const-correctness)

        using policy_t = Kokkos::RangePolicy<Exec>;
        using index_t  = typename policy_t::index_type;

        Kokkos::parallel_for(
            Kokkos::RangePolicy(exec, 0, mat.numRows()),
            KOKKOS_LAMBDA(const index_t irow)
            {
                using local_ordinal_t = typename MatrixType::ordinal_type;

                const auto row = mat.rowConst(irow);

                for(local_ordinal_t ival = 0; ival < row.length; ++ival)
                {
                    if(static_cast<index_t>(row.colidx(ival)) == irow)
                    {
                        tmp(irow) = 1. / row.value(ival);
                        break;
                    }
                }
            }
        );

        values = std::move(tmp);
    }

    template <typename DstType, typename SrcType>
    struct Apply
    {
        DstType dst;
        typename SrcType::const_type src;
        const_values_view_t values;

        template <std::integral T>
        KOKKOS_FUNCTION
        void operator()(const T irow) const {
            dst(irow) = values(irow) * src(irow);
        }
    };

    template <
        Kokkos::ExecutionSpace Exec,
        typename DstType,
        typename SrcType
    > requires (std::remove_cvref_t<DstType>::rank() == 1 && std::remove_cvref_t<SrcType>::rank() == 1)
    void apply(const Exec& exec, DstType&& dst, SrcType&& src) const
    {
        Kokkos::parallel_for(
            Kokkos::RangePolicy(exec, 0, values.size()),
            Apply<std::remove_cvref_t<DstType>, std::remove_cvref_t<SrcType>>{.dst = std::forward<DstType>(dst), .src = std::forward<SrcType>(src), .values = values}
        );
    }

    template <
        typename Pred,
        typename DstType,
        typename SrcType
    > requires (std::remove_cvref_t<DstType>::rank() == 1 && std::remove_cvref_t<SrcType>::rank() == 1 && IsGraphNode<std::remove_cvref_t<Pred>>)
    decltype(auto) apply(const Pred& pred, DstType&& dst, SrcType&& src) const {
        using execution_space = typename std::remove_cvref_t<Pred>::execution_space;

        return pred.then_parallel_for(
            Kokkos::RangePolicy<execution_space>(0, values.size()),
            Apply<std::remove_cvref_t<DstType>, std::remove_cvref_t<SrcType>>{.dst = std::forward<DstType>(dst), .src = std::forward<SrcType>(src), .values = values}
        );
    }
};

/**
 * @brief Based on https://en.wikipedia.org/wiki/Jacobi_method.
 *
 * @note According to https://en.wikipedia.org/wiki/Jacobi_method#Convergence,
 *       a sufficient condition for the convergence is that the matrix is diagonally dominant.
 */
template <typename MatrixType>
struct JacobiPreconditioner
{
    using const_mat_t = typename MatrixType::const_type;
    using tmp_t       = typename MatrixType::values_type::non_const_type;

    using local_ordinal_t = typename MatrixType::ordinal_type;

    const_mat_t mat;
    tmp_t tmp;

    using sweep_t = unsigned short int;

    sweep_t num_sweeps = 4; //! Number of sweeps.

    template <Kokkos::ExecutionSpace Exec>
    JacobiPreconditioner(const Exec& exec, const_mat_t mat_)
        : mat{std::move(mat_)},
          tmp(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Jacobi temporary storage", exec), mat.numRows())
    {}

    template <typename DstType, typename SrcType>
    struct ApplyFirstPass
    {
        typename DstType::non_const_type dst;
        typename SrcType::const_type src;
        const_mat_t mat;

        template <std::integral T>
        KOKKOS_FUNCTION
        void operator()(const T irow) const
        {
            const auto row = mat.rowConst(irow);

            for(local_ordinal_t ival = 0; ival < row.length; ++ival)
            {
                if(static_cast<T>(row.colidx(ival)) == irow)
                {
                    dst(irow) = src(irow) / row.value(ival);
                    break;
                }
            }
        }
    };

    template <typename DstType, typename SrcType>
    struct Apply
    {
        typename DstType::const_type dst;
        typename SrcType::const_type src;
        const_mat_t mat;
        tmp_t tmp;

        static_assert(std::same_as<typename DstType::non_const_value_type, typename SrcType::non_const_value_type>);
        static_assert(std::same_as<typename DstType::non_const_value_type, typename const_mat_t::non_const_value_type>);

        using value_t = typename DstType::non_const_value_type;

        template <std::integral T>
        KOKKOS_FUNCTION
        void operator()(const T irow) const
        {
            value_t accu = 0., diag = 0.;

            const auto row = mat.rowConst(irow);

            for(local_ordinal_t ival = 0; ival < row.length; ++ival)
            {
                const auto icol = static_cast<T>(row.colidx(ival));

                if(static_cast<T>(icol) == irow) {
                    diag = row.value(ival);
                } else {
                    accu += row.value(ival) * dst(icol);
                }
            }

            tmp(irow) = (src(irow) - accu) / diag;
        }
    };

    template <Kokkos::ExecutionSpace Exec, Kokkos::utils::concepts::ViewOfRank<1> DstType, Kokkos::utils::concepts::ViewOfRank<1> SrcType>
    void apply(const Exec& exec, const DstType& dst, const SrcType& src) const
    {
        /// In the first pass, we should set @p dst to 0. But as it is equivalent to diagonal
        /// scaling, we can specialize the kernel for the first pass and save one deep copy operation.
        Kokkos::parallel_for(
            Kokkos::RangePolicy(exec, 0, mat.numRows()),
            ApplyFirstPass<DstType, SrcType>{.dst = dst, .src = src, .mat = mat}
        );
        for(sweep_t isweep = 1; isweep < num_sweeps; ++isweep)
        {
            Kokkos::parallel_for(
                Kokkos::RangePolicy(exec, 0, mat.numRows()),
                Apply<DstType, SrcType>{.dst = dst, .src = src, .mat = mat, .tmp = tmp}
            );
            Kokkos::deep_copy(exec, dst, tmp);
        }
    }

    template <typename Pred, typename DstType, typename SrcType> requires IsGraphNode<std::remove_cvref_t<Pred>>
    decltype(auto) apply(const Pred& pred, const DstType& dst, const SrcType& src) const
    {
        using execution_space  = typename std::remove_cvref_t<Pred>::execution_space;
        using graph_node_ref_t = Kokkos::Experimental::GraphNodeRef<execution_space>;

        /// In the first pass, we should set @p dst to 0. But as it is equivalent to diagonal
        /// scaling, we can specialize the kernel for the first pass and save one deep copy operation.
        graph_node_ref_t next = pred.then_parallel_for(
            Kokkos::RangePolicy<execution_space>(0, mat.numRows()),
            ApplyFirstPass<DstType, SrcType>{.dst = dst, .src = src, .mat = mat}
        );
        for(sweep_t isweep = 1; isweep < num_sweeps; ++isweep)
        {
            auto next_in_loop = next.then_parallel_for(
                Kokkos::RangePolicy<execution_space>(0, mat.numRows()),
                Apply<DstType, SrcType>{.dst = dst, .src = src, .mat = mat, .tmp = tmp}
            );
            next = then_deep_copy(next_in_loop, dst, tmp);
        }
        return next;
    }
};

} // namespace algorithms::pcg

#endif // GRAPH_DISPATCHING_ALGORITHMS_PCG_PRECONDITIONERS_HPP
