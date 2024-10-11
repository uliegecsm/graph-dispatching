#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_HPP

namespace Kokkos::Experimental::graph::details
{

/**
 * @brief Helper for piping support.
 *
 * Before the @c operator| is called, we have a *partial* algorithm because we
 * don't know the *parent* yet.
 *
 * @note It must be specialized for each @c Kokkos parallel tag.
 */
template <typename Tag, typename ...>
struct PartialAlgorithm;

} // Kokkos::Experimental::graph::details

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_HPP
