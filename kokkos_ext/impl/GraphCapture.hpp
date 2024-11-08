#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCAPTURE_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCAPTURE_HPP

#include <concepts>

namespace Kokkos::Experimental::graph
{

//! Enable stream capture if @p exec is a @ref StatefulGraphNode.
// template <typename Exec, typename Closure>
// void capture(const Exec& exec, Closure&& closure)
// {
//     std::forward<Closure>(closure)();
// }

// template <typename Exec, typename Closure> requires std::same_as<std::remove_cvref_t<Exec>, StatefulGraphNode>
// void capture(const )
// {
//     if constexpr (std::same_as<std::remove_cvref_t<Exec>, StatefulGraphNode>) {
//         CHECK_CALL(PREFIXED_API(StreamBeginCapture)(get_stream(exec).stream, PREFIXED_API(StreamCaptureModeGlobal)));
//     }

//     std::forward<Closure>(closure)();

//     if constexpr (std::same_as<std::remove_cvref_t<Exec>, StatefulGraphNode>)
//     {
//         Graph library(nullptr);

//         CHECK_CALL(PREFIXED_API(StreamEndCapture)(get_stream(exec).stream, &library.graph));

//         [[maybe_unused]] const auto library_as_node = library.add(exec.graph, {exec.node});
//     }
// }

} // namespace Kokkos::Experimental::graph

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCAPTURE_HPP
