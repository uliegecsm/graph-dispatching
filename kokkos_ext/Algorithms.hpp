#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_ALGORITHMS_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_ALGORITHMS_HPP

#include <concepts>

#include "Kokkos_Core.hpp"

namespace KokkosExt
{

//! User-provided generator. See https://github.com/kokkos/kokkos/issues/4498.
template <typename ValueType>
struct FillSequence
{
    ValueType offset;

    template <std::integral T>
    KOKKOS_FUNCTION
    constexpr auto operator()(const T index) const { return offset + static_cast<ValueType>(index); }
};

namespace Impl
{
//! Generate at index utility, see https://github.com/kokkos/kokkos/issues/4498.
template <typename ViewType, typename Generator>
struct GenerateAt
{
    ViewType data;
    Generator gen;

    template <std::integral T>
    KOKKOS_FUNCTION
    constexpr void operator()(const T index) const {
        data(index) = gen(index);
    }
};
}

/// Some principle as @c std::generate, but the generator takes the index and all elements
/// of the sequence can be initialized independantly.
/// See also https://github.com/kokkos/kokkos/issues/4498.
template <typename Label, typename Exec, typename ViewType, typename Generator>
void generate_at(Label&& label, Exec&& exec, ViewType&& data, Generator&& gen)
{
    static_assert(std::remove_cvref_t<ViewType>::rank() == 1, "Rank-1 view required.");

    static_assert(std::is_invocable_r<
        typename std::remove_cvref_t<ViewType>::value_type,
        Generator,
        typename std::remove_cvref_t<Exec>::size_type
    >::value);

    Kokkos::parallel_for(
        std::forward<Label>(label),
        Kokkos::RangePolicy(std::forward<Exec>(exec), 0, data.size()),
        Impl::GenerateAt{.data = std::forward<ViewType>(data), .gen = std::forward<Generator>(gen)}
    );
}

//! Fill @p data as @c std::iota would do, but assuming each element can be initialized independantly.
template <typename Exec, typename ViewType, typename ValueType = typename std::remove_cvref_t<ViewType>::value_type> requires (std::remove_cvref_t<ViewType>::rank() == 1)
void fill_sequence(Exec&& exec, ViewType&& data, ValueType&& value)
{
    generate_at("iota", std::forward<Exec>(exec), std::forward<ViewType>(data), FillSequence{.offset = std::forward<ValueType>(value)});
};
} // namespace KokkosExt

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_ALGORITHMS_HPP
