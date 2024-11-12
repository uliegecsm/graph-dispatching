// #ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCAPTURE_HPP
// #define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCAPTURE_HPP

// #include <concepts>

// #include "Kokkos_Core.hpp"

// namespace Kokkos::Experimental::graph
// {
// //! Add a task.
// template <typename Exec, typename Task, typename Data>
// decltype(auto) task(Exec&& exec, Task&& task, Data&& data);

// //! Specialization when @p exec is not a graph. We get "eager execution".
// template <typename Exec, typename Task, typename Data> requires Kokkos::is_execution_space_v<Exec>
// decltype(auto) task(const Exec& exec, const Task& task, Data&& data)
// {
//     task(exec, std::forward<Data>(data));

//     //! If data is given, we have guarenteed that it would be kept alive until completion.
//     // if constexpr (data not empty and not moved) exec.fence();
//     return exec;
// }

// // //! Specialization when @p exec is a @c Kokkos defaulted graph (@c Serial or @c OpenMP for instance).
// // template <typename Exec, typename Task, typename Data> requires ( defaulted graph exec )
// // decltype(auto) Kokkos::graph::task(const Exec& exec, Task&& task, Data&& data)
// // {
// //     /// 1. Create graph node with @p task and @p data. Add it to graph.
// //     ///    Since @p data is grafted it to graph, its lifetime is bounded to the one of graph (and thereby to the node's lifetime).
// //     return node;
// // }

// //! Specialization when @p exec is a @c Kokkos specialized graph (@c Cuda or @c HIP for instance).
// template <typename Exec, typename Task, typename Data>
// decltype(auto) task(const Exec& exec, Task&& task, Data&& data)
// {
//     /// 1. Create graph node(s) by executing @p task and using stream capture.
//     decltype(auto) stream = get_exec(exec).cuda_stream();

//     KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

//     std::forward<Task>(task)(exec, data);

//     cudaGraph_t library(nullptr);

//     CHECK_CALL(cudaStreamEndCapture(stream, &library));

//     auto library_as_node = exec | Kokkos::Experimental::graph::then(library);

//     /// 2. Create data handle with @p data. Graft it to graph to bind its lifetime to the one of graph.
//     /// @todo

//     return library_as_node;
// }

// } // namespace Kokkos::Experimental::graph

// #endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCAPTURE_HPP
