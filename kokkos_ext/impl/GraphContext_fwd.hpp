#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCONTEXT_FWD_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCONTEXT_FWD_HPP

#include "Kokkos_Concepts.hpp"

namespace Kokkos::Experimental::details::graph {

template <typename Exec>
requires Kokkos::is_execution_space_v<Exec>
struct GraphScheduler;

} // namespace Kokkos::Experimental::details::graph

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCONTEXT_FWD_HPP
