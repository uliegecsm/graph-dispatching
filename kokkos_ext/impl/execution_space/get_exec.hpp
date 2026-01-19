#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_GET_EXEC_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_GET_EXEC_HPP

namespace Kokkos::Experimental::details::execution_space {

/**
 * Query an object for its @c Kokkos execution space instance.
 *
 * See also https://github.com/NVIDIA/cccl/blob/6e592beda9c50aeb3cc62dd1036d509f540ccbe7/libcudacxx/include/cuda/__stream/get_stream.h.
 */
struct get_exec_t {
    template <stdexec::__member_queryable_with<get_exec_t> Env>
    [[nodiscard]]
    constexpr auto operator()(const Env& env) const noexcept {
        return env.query(*this);
    }

    template <typename Env>
    requires(!stdexec::__member_queryable_with<Env, get_exec_t>) && requires(const Env& env) { env.exec; }
    [[nodiscard]]
    constexpr auto operator()(const Env& env) const noexcept {
        return env.exec;
    }

    [[nodiscard]]
    constexpr bool query(stdexec::forwarding_query_t) const noexcept {
        return true;
    }
};

inline constexpr get_exec_t get_exec{};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_GET_EXEC_HPP
