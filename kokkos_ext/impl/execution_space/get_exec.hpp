#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_GET_EXEC_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_GET_EXEC_HPP

namespace Kokkos::Experimental::details::execution_space {

/**
 * Query an object for its @c Kokkos execution space instance.
 *
 * See also https://github.com/NVIDIA/cccl/blob/6e592beda9c50aeb3cc62dd1036d509f540ccbe7/libcudacxx/include/cuda/__stream/get_stream.h.
 */
struct get_exec_t
    : public ::stdexec::__query<get_exec_t>
    , ::stdexec::forwarding_query_t {
    using ::stdexec::__query<get_exec_t>::operator();
};

inline constexpr get_exec_t get_exec{};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_GET_EXEC_HPP
