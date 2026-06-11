#ifndef GRAPH_DISPATCHING_ALGORITHMS_PCG_PRECONDITIONERS_HPP
#define GRAPH_DISPATCHING_ALGORITHMS_PCG_PRECONDITIONERS_HPP

#include <concepts>
#include <optional>

#include "Kokkos_Graph.hpp"

#include "kokkos-utils/concepts/View.hpp"
#include "kokkos-utils/impl/type_traits.hpp"

#include "algorithms/cg/Helpers.hpp"

template <typename T>
concept IsGraphNode = Kokkos::utils::impl::InstanceOf<T, Kokkos::Experimental::GraphNodeRef>;

namespace algorithms::pcg {

/**
 * @brief Return the uniform byte fill value if all bytes of @p value are identical
 *        and the type is trivially copyable, @c std::nullopt otherwise.
 *
 * Inspired by https://github.com/kokkos/kokkos/blob/225173380476c6f22b08e79d8626a7c918a512e0/core/src/Kokkos_CopyViews.hpp#L899.
 */
template <typename T>
std::optional<unsigned char> uniform_byte_fill(const T& value) noexcept {
    if constexpr (!std::is_trivially_copyable_v<T>)
        return std::nullopt;

    //! Single-byte types can be converted directly.
    if constexpr (sizeof(T) == 1)
        return *reinterpret_cast<const unsigned char*>(&value);

    //! A value-initialisation of a scalar type always produces an all-zero bit pattern.
    if constexpr (std::is_scalar_v<T>)
        if (value == T())
            return (unsigned char) 0;

    //! For types with unique object representations, inspect the byte pattern.
    if constexpr (std::is_standard_layout_v<T> && std::has_unique_object_representations_v<T>) {
        const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
        for (std::size_t i = 1; i < sizeof(T); ++i)
            if (bytes[i] != bytes[0])
                return std::nullopt;
        return bytes[0];
    }

    return std::nullopt;
}

template <Kokkos::ExecutionSpace Exec, typename, typename>
struct DeepCopy;

template <Kokkos::ExecutionSpace Exec, typename>
struct Memset;

//! Specialization for copying a rank-1 view into another rank-1 view.
template <
    Kokkos::ExecutionSpace Exec,
    Kokkos::utils::concepts::ViewOfRank<1> DstType,
    Kokkos::utils::concepts::ViewOfRank<1> SrcType
>
struct DeepCopy<Exec, DstType, SrcType> {
    typename DstType::non_const_type dst;
    typename SrcType::const_type src;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T index) const noexcept {
        dst(index) = src(index);
    }

    template <typename Pred>
    auto add(Pred&& pred, const Kokkos::Impl::DeviceHandle<Exec>& device_handle) && {
        const auto size = dst.size();
        if (size != src.size())
            Kokkos::abort("Views must be of the same size.");
        return std::forward<Pred>(pred).then_parallel_for(
            Kokkos::Experimental::node_props(device_handle, "deep copy"),
            Kokkos::RangePolicy<Exec>(0, size),
            std::move(*this));
    }
};

//! Specialization for copying a scalar into a rank-1 view.
template <Kokkos::ExecutionSpace Exec, Kokkos::utils::concepts::ViewOfRank<1> DstType, typename SrcType>
requires std::is_scalar_v<SrcType>
struct DeepCopy<Exec, DstType, SrcType> {
    typename DstType::non_const_type dst;
    SrcType src;

    static_assert(std::convertible_to<SrcType, typename DstType::non_const_value_type>);
    static_assert(std::is_trivially_copyable_v<SrcType>);

    //! Fallback when memset is not applicable.
    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T index) const noexcept {
        dst(index) = src;
    }

    //! For views whose span is contiguous, use @ref uniform_byte_fill to determine if @ref Memset is applicable.
    template <typename Pred>
    auto
        add(Pred&& pred, const Kokkos::Impl::DeviceHandle<Exec>& device_handle) && -> Kokkos::Experimental::GraphNodeRef<
            Exec,
            Kokkos::Experimental::TypeErasedTag,
            Kokkos::Experimental::TypeErasedTag
        > {
        if (dst.span_is_contiguous()) {
            if (const auto fill_byte = uniform_byte_fill(src); fill_byte.has_value()) {
                return Memset<Exec, DstType>{.dst = std::move(dst), .src = *fill_byte}
                    .add(std::forward<Pred>(pred), device_handle);
            }
        }
        const auto size = dst.size();
        return std::forward<Pred>(pred).then_parallel_for(
            Kokkos::Experimental::node_props(device_handle, "deep copy"),
            Kokkos::RangePolicy<Exec>(0, size),
            std::move(*this));
    }
};

template <Kokkos::ExecutionSpace Exec, Kokkos::utils::concepts::ViewOfRank<1> DstType>
struct Memset<Exec, DstType> {
    typename DstType::non_const_type dst;
    unsigned char src;

    void operator()() const {
        std::memset(dst.data(), src, sizeof(typename DstType::non_const_value_type) * dst.size());
    }

    template <typename Pred>
    auto add(Pred&& pred, const Kokkos::Impl::DeviceHandle<Exec>& device_handle) && {
        return std::forward<Pred>(pred).then_host(Kokkos::Experimental::node_props(device_handle), std::move(*this));
    }
};

#if defined(KOKKOS_ENABLE_CUDA)
template <Kokkos::utils::concepts::ViewOfRank<1> DstType, Kokkos::utils::concepts::ViewOfRank<1> SrcType>
struct DeepCopy<Kokkos::Cuda, DstType, SrcType> {
    typename DstType::non_const_type dst;
    typename SrcType::const_type src;

    static_assert(std::same_as<typename DstType::value_type, typename SrcType::value_type>);

    void operator()(const Kokkos::Cuda& exec) const {
        if (dst.size() != src.size())
            Kokkos::abort("Views must be of the same size.");
        if (!dst.span_is_contiguous() || !src.span_is_contiguous())
            Kokkos::abort("Only contiguous spans are supported for CUDA memcpy.");
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemcpyAsync(
            dst.data(),
            src.data(),
            sizeof(typename DstType::value_type) * dst.size(),
            cudaMemcpyDefault,
            exec.cuda_stream()));
    }

    template <typename Pred>
    auto add(Pred&& pred, const Kokkos::Impl::DeviceHandle<Kokkos::Cuda>& device_handle) && {
        return std::forward<Pred>(pred).cuda_capture(device_handle.m_exec, std::move(*this));
    }
};

template <Kokkos::utils::concepts::ViewOfRank<1> DstType>
struct Memset<Kokkos::Cuda, DstType> {
    typename DstType::non_const_type dst;
    unsigned char src;

    void operator()(const Kokkos::Cuda& exec) const {
        if (!dst.span_is_contiguous())
            Kokkos::abort("Only contiguous spans are supported for CUDA memset.");
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemsetAsync(
            dst.data(), src, sizeof(typename DstType::non_const_value_type) * dst.size(), exec.cuda_stream()));
    }

    template <typename Pred>
    auto add(Pred&& pred, const Kokkos::Impl::DeviceHandle<Kokkos::Cuda>& device_handle) && {
        return std::forward<Pred>(pred).cuda_capture(device_handle.m_exec, std::move(*this));
    }
};
#elif defined(KOKKOS_ENABLE_HIP) && (HIP_VERSION_MAJOR >= 7 && HIP_VERSION_MINOR >= 2)
template <Kokkos::utils::concepts::ViewOfRank<1> DstType, Kokkos::utils::concepts::ViewOfRank<1> SrcType>
struct DeepCopy<Kokkos::HIP, DstType, SrcType> {
    typename DstType::non_const_type dst;
    typename SrcType::const_type src;

    static_assert(std::same_as<typename DstType::value_type, typename SrcType::value_type>);

    void operator()(const Kokkos::HIP& exec) const {
        if (dst.size() != src.size())
            Kokkos::abort("Views must be of the same size.");
        if (!dst.span_is_contiguous() || !src.span_is_contiguous())
            Kokkos::abort("Only contiguous spans are supported for HIP memcpy.");
        KOKKOS_IMPL_HIP_SAFE_CALL(hipMemcpyAsync(
            dst.data(),
            src.data(),
            sizeof(typename DstType::value_type) * dst.size(),
            hipMemcpyDefault,
            exec.hip_stream()));
    }

    template <typename Pred>
    auto add(Pred&& pred, const Kokkos::Impl::DeviceHandle<Kokkos::HIP>& device_handle) && {
        return std::forward<Pred>(pred).hip_capture(device_handle.m_exec, std::move(*this));
    }
};

template <Kokkos::utils::concepts::ViewOfRank<1> DstType>
struct Memset<Kokkos::HIP, DstType> {
    typename DstType::non_const_type dst;
    unsigned char src;

    void operator()(const Kokkos::HIP& exec) const {
        if (!dst.span_is_contiguous())
            Kokkos::abort("Only contiguous spans are supported for HIP memset.");
        KOKKOS_IMPL_HIP_SAFE_CALL(hipMemsetAsync(
            dst.data(), src, sizeof(typename DstType::non_const_value_type) * dst.size(), exec.hip_stream()));
    }

    template <typename Pred>
    auto add(Pred&& pred, const Kokkos::Impl::DeviceHandle<Kokkos::HIP>& device_handle) && {
        return std::forward<Pred>(pred).hip_capture(device_handle.m_exec, std::move(*this));
    }
};
#elif defined(KOKKOS_ENABLE_SYCL) && defined(KOKKOS_IMPL_SYCL_GRAPH_SUPPORT)
template <Kokkos::utils::concepts::ViewOfRank<1> DstType, Kokkos::utils::concepts::ViewOfRank<1> SrcType>
struct DeepCopy<Kokkos::SYCL, DstType, SrcType> {
    typename DstType::non_const_type dst;
    typename SrcType::const_type src;

    static_assert(std::same_as<typename DstType::value_type, typename SrcType::value_type>);

    void operator()(const Kokkos::SYCL& exec) const {
        if (!dst.span_is_contiguous() || !src.span_is_contiguous())
            Kokkos::abort("Only contiguous spans are supported for SYCL memcpy.");
        exec.sycl_queue().memcpy(dst.data(), src.data(), sizeof(typename DstType::value_type) * dst.size());
    }

    template <typename Pred>
    auto add(Pred&& pred, const Kokkos::Impl::DeviceHandle<Kokkos::SYCL>& device_handle) && {
        return std::forward<Pred>(pred).sycl_capture(device_handle.m_exec, std::move(*this));
    }
};

template <Kokkos::utils::concepts::ViewOfRank<1> DstType>
struct Memset<Kokkos::SYCL, DstType> {
    typename DstType::non_const_type dst;
    unsigned char src;

    void operator()(const Kokkos::SYCL& exec) const {
        if (!dst.span_is_contiguous())
            Kokkos::abort("Only contiguous spans are supported for SYCL memset.");
        exec.sycl_queue().memset(dst.data(), src, sizeof(typename DstType::non_const_value_type) * dst.size());
    }

    template <typename Pred>
    auto add(Pred&& pred, const Kokkos::Impl::DeviceHandle<Kokkos::SYCL>& device_handle) && {
        return std::forward<Pred>(pred).sycl_capture(device_handle.m_exec, std::move(*this));
    }
};
#endif

//! @todo Use a proper @c memcpy node.
template <typename Pred, typename DstType, typename SrcType>
requires(Kokkos::utils::concepts::ViewOfRank<std::remove_cvref_t<DstType>, 1>)
decltype(auto) then_deep_copy(Pred&& pred, DstType&& dst, SrcType&& src) {
    using execution_space = typename std::remove_cvref_t<Pred>::execution_space;
    using deep_copy_t = DeepCopy<execution_space, std::remove_cvref_t<DstType>, std::remove_cvref_t<SrcType>>;

    return deep_copy_t{.dst = std::forward<DstType>(dst), .src = std::forward<SrcType>(src)}
        .add(std::forward<Pred>(pred), Kokkos::Impl::GraphAccess::get_node_ptr(pred)->get_device_handle());
}

//! Identity.
template <typename MatrixType>
struct IdentityPreconditioner {
    template <Kokkos::ExecutionSpace Exec>
    IdentityPreconditioner(const Exec&, const MatrixType&) {
    }

    //! @note It must be a deep copy because it's not supposed to touch the address of @p dst.
    template <Kokkos::ExecutionSpace Exec, Kokkos::utils::concepts::ViewOfRank<1> VectorType>
    void apply(const Exec& exec, const VectorType& dst, const VectorType& src) const {
        Kokkos::deep_copy(exec, dst, src);
    }

    template <typename Pred, typename DstType, typename SrcType>
    requires IsGraphNode<std::remove_cvref_t<Pred>>
    decltype(auto) apply(const Pred& pred, DstType&& dst, SrcType&& src) const {
        return then_deep_copy(pred, std::forward<DstType>(dst), std::forward<SrcType>(src));
    }
};

//! Diagonal scaling. Simple, but not very efficient in general.
template <typename MatrixType>
struct DiagonalPreconditioner {
    using values_view_t = typename MatrixType::values_type::non_const_type;
    using const_values_view_t = typename values_view_t::const_type;

    //! The constructor will pre-compute the inverse of the diagonal values.
    const_values_view_t values;

    template <Kokkos::ExecutionSpace Exec>
    DiagonalPreconditioner(const Exec& exec, const typename MatrixType::const_type& mat) {
        this->init(exec, mat);
    }

    template <typename Exec>
    void init(const Exec& exec, const typename MatrixType::const_type& mat) {
        values_view_t // NOLINT(misc-const-correctness)
            tmp(Kokkos::view_alloc(Kokkos::WithoutInitializing, "inverse of diagonal values", exec), mat.numRows());

        using policy_t = Kokkos::RangePolicy<Exec>;
        using index_t = typename policy_t::index_type;

        Kokkos::parallel_for(
            Kokkos::RangePolicy(exec, 0, mat.numRows()), KOKKOS_LAMBDA(const index_t irow) {
                using local_ordinal_t = typename MatrixType::ordinal_type;

                const auto row = mat.rowConst(irow);

                for (local_ordinal_t ival = 0; ival < row.length; ++ival) {
                    if (static_cast<index_t>(row.colidx(ival)) == irow) {
                        tmp(irow) = 1. / row.value(ival);
                        break;
                    }
                }
            });

        values = std::move(tmp);
    }

    template <typename DstType, typename SrcType>
    struct Apply {
        DstType dst;
        typename SrcType::const_type src;
        const_values_view_t values;

        template <std::integral T>
        KOKKOS_FUNCTION void operator()(const T irow) const {
            dst(irow) = values(irow) * src(irow);
        }
    };

    template <Kokkos::ExecutionSpace Exec, typename DstType, typename SrcType>
    requires(std::remove_cvref_t<DstType>::rank() == 1 && std::remove_cvref_t<SrcType>::rank() == 1)
    void apply(const Exec& exec, DstType&& dst, SrcType&& src) const {
        Kokkos::parallel_for(
            Kokkos::RangePolicy(exec, 0, values.size()),
            Apply<std::remove_cvref_t<DstType>, std::remove_cvref_t<SrcType>>{
                .dst = std::forward<DstType>(dst), .src = std::forward<SrcType>(src), .values = values});
    }

    template <typename Pred, typename DstType, typename SrcType>
    requires(
        std::remove_cvref_t<DstType>::rank() == 1 && std::remove_cvref_t<SrcType>::rank() == 1
        && IsGraphNode<std::remove_cvref_t<Pred>>)
    decltype(auto) apply(const Pred& pred, DstType&& dst, SrcType&& src) const {
        using execution_space = typename std::remove_cvref_t<Pred>::execution_space;

        return pred.then_parallel_for(
            Kokkos::RangePolicy<execution_space>(0, values.size()),
            Apply<std::remove_cvref_t<DstType>, std::remove_cvref_t<SrcType>>{
                .dst = std::forward<DstType>(dst), .src = std::forward<SrcType>(src), .values = values});
    }
};

/**
 * @brief Based on https://en.wikipedia.org/wiki/Jacobi_method.
 *
 * @note According to https://en.wikipedia.org/wiki/Jacobi_method#Convergence,
 *       a sufficient condition for the convergence is that the matrix is diagonally dominant.
 */
template <typename MatrixType>
struct JacobiPreconditioner {
    using const_mat_t = typename MatrixType::const_type;
    using tmp_t = typename MatrixType::values_type::non_const_type;

    using local_ordinal_t = typename MatrixType::ordinal_type;

    const_mat_t mat;
    tmp_t tmp;

    using sweep_t = unsigned short int;

    sweep_t num_sweeps = 4; //! Number of sweeps.

    template <Kokkos::ExecutionSpace Exec>
    JacobiPreconditioner(const Exec& exec, const_mat_t mat_)
        : mat{std::move(mat_)}
        , tmp(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Jacobi temporary storage", exec), mat.numRows()) {
    }

    template <typename DstType, typename SrcType>
    struct ApplyFirstPass {
        typename DstType::non_const_type dst;
        typename SrcType::const_type src;
        const_mat_t mat;

        template <std::integral T>
        KOKKOS_FUNCTION void operator()(const T irow) const {
            const auto row = mat.rowConst(irow);

            for (local_ordinal_t ival = 0; ival < row.length; ++ival) {
                if (static_cast<T>(row.colidx(ival)) == irow) {
                    dst(irow) = src(irow) / row.value(ival);
                    break;
                }
            }
        }
    };

    template <typename DstType, typename SrcType>
    struct Apply {
        typename DstType::const_type dst;
        typename SrcType::const_type src;
        const_mat_t mat;
        tmp_t tmp;

        static_assert(std::same_as<typename DstType::non_const_value_type, typename SrcType::non_const_value_type>);
        static_assert(std::same_as<typename DstType::non_const_value_type, typename const_mat_t::non_const_value_type>);

        using value_t = typename DstType::non_const_value_type;

        template <std::integral T>
        KOKKOS_FUNCTION void operator()(const T irow) const {
            value_t accu = 0., diag = 0.;

            const auto row = mat.rowConst(irow);

            for (local_ordinal_t ival = 0; ival < row.length; ++ival) {
                const auto icol = static_cast<T>(row.colidx(ival));

                if (static_cast<T>(icol) == irow) {
                    diag = row.value(ival);
                } else {
                    accu += row.value(ival) * dst(icol);
                }
            }

            tmp(irow) = (src(irow) - accu) / diag;
        }
    };

    template <
        Kokkos::ExecutionSpace Exec,
        Kokkos::utils::concepts::ViewOfRank<1> DstType,
        Kokkos::utils::concepts::ViewOfRank<1> SrcType
    >
    void apply(const Exec& exec, const DstType& dst, const SrcType& src) const {
        /// In the first pass, we should set @p dst to 0. But as it is equivalent to diagonal
        /// scaling, we can specialize the kernel for the first pass and save one deep copy operation.
        Kokkos::parallel_for(
            Kokkos::RangePolicy(exec, 0, mat.numRows()),
            ApplyFirstPass<DstType, SrcType>{.dst = dst, .src = src, .mat = mat});

        /// For @ref num_sweeps larger than one, ping-pong between @p dst and @ref tmp
        /// to avoid per-sweep deep copies.
        for (sweep_t isweep = 1; isweep < num_sweeps; ++isweep) {
            if (isweep & 1) {
                Kokkos::parallel_for(
                    Kokkos::RangePolicy(exec, 0, mat.numRows()),
                    Apply<DstType, SrcType>{.dst = dst, .src = src, .mat = mat, .tmp = tmp});
            } else {
                Kokkos::parallel_for(
                    Kokkos::RangePolicy(exec, 0, mat.numRows()),
                    Apply<DstType, SrcType>{.dst = tmp, .src = src, .mat = mat, .tmp = dst});
            }
        }

        /// If @ref num_sweeps is even, the last write landed in @ref tmp, copy back to @p dst.
        if (num_sweeps > 1 && !(num_sweeps & 1))
            Kokkos::deep_copy(exec, dst, tmp);
    }

    template <typename Pred, typename DstType, typename SrcType>
    requires IsGraphNode<std::remove_cvref_t<Pred>>
    decltype(auto) apply(Pred&& pred, const DstType& dst, const SrcType& src) const {
        using execution_space = typename std::remove_cvref_t<Pred>::execution_space;
        using graph_node_ref_t = Kokkos::Experimental::GraphNodeRef<execution_space>;

        /// In the first pass, we should set @p dst to 0. But as it is equivalent to diagonal
        /// scaling, we can specialize the kernel for the first pass and save one deep copy operation.
        graph_node_ref_t next = std::forward<Pred>(pred).then_parallel_for(
            Kokkos::RangePolicy<execution_space>(0, mat.numRows()),
            ApplyFirstPass<DstType, SrcType>{.dst = dst, .src = src, .mat = mat});

        /// For @ref num_sweeps larger than one, ping-pong between @p dst and @ref tmp
        /// to avoid per-sweep deep copies.
        for (sweep_t isweep = 1; isweep < num_sweeps; ++isweep) {
            if (isweep & 1) {
                next = next.then_parallel_for(
                    Kokkos::RangePolicy<execution_space>(0, mat.numRows()),
                    Apply<DstType, SrcType>{.dst = dst, .src = src, .mat = mat, .tmp = tmp});
            } else {
                next = next.then_parallel_for(
                    Kokkos::RangePolicy<execution_space>(0, mat.numRows()),
                    Apply<DstType, SrcType>{.dst = tmp, .src = src, .mat = mat, .tmp = dst});
            }
        }

        /// If @ref num_sweeps is even, the last write landed in @ref tmp, copy back to @p dst.
        if (num_sweeps > 1 && !(num_sweeps & 1))
            next = then_deep_copy(next, dst, tmp);

        return next;
    }
};

} // namespace algorithms::pcg

#endif // GRAPH_DISPATCHING_ALGORITHMS_PCG_PRECONDITIONERS_HPP
