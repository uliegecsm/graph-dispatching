#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_KOKKOS_RANGEPOLICY_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_KOKKOS_RANGEPOLICY_HPP

#include <concepts>

namespace Kokkos::Experimental::graph
{
/**
 * @brief Range policy bounds.
 *
 * This policy is a replacer for the @c Kokkos::RangePolicy. It is needed because
 * @c Kokkos::RangePolicy mixes the policy bounds and the execution space instance,
 * and we currently have no mean to determine if a @c Kokkos::RangePolicy has been built
 * with an execution space instance or not.
 */
template <std::integral T = typename Kokkos::DefaultExecutionSpace::memory_space::size_type>
struct RangePolicy
{
    T m_begin, m_end;

    RangePolicy(const T begin, const T end) : m_begin(begin), m_end(end) {}

    //! @name Getters similar to those of @c Kokkos::RangePolicy.
    ///@{
    auto begin() const { return m_begin; }
    auto end()   const { return m_end; }
    ///@}
};

} // namespace Kokkos::Experimental::graph

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_KOKKOS_RANGEPOLICY_HPP
