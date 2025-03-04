#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_CONCEPTS_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_CONCEPTS_HPP

#include <concepts>

namespace Kokkos::Experimental
{
/**
 * @brief Weaker concept than @c stdexec::environment_provider.
 *
 * Reference:
 *  - https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/stdexec/__detail/__env.hpp#L656-L660
 */
template <class T>
concept environment_provider = requires (const T& arg) {
    { arg.get_env() };
};

/**
 * @brief Subset of @c std::execution::sender.
 *
 * References:
 *  - https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/stdexec/__detail/__senders_core.hpp#L35-L52
 */
template <typename T>
concept sender = std::move_constructible<T> && std::copy_constructible<T> && environment_provider<T>;

//! @todo Make it real.
template <typename T>
concept receiver = requires (T&& arg) {
    { std::move(arg).set_value() } -> std::same_as<void>;
};

//! @todo Make it real.
template <typename T>
concept operation_state = true;

//! @todo Make it real.
template <typename T>
concept scheduler = true;

namespace graph::details
{
/// @todo This constraint ain't clear but should support the case of chaining
///       without an underlying @c Kokkos::Graph.
template <typename Sender>
concept is_graph_sender = ! Kokkos::is_execution_space_v<Sender>;

} // namespace graph::details

} // namespace Kokkos::Experimental

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_CONCEPTS_HPP
