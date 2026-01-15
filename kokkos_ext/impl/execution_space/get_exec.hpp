#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_QUERIES_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_QUERIES_HPP

namespace Kokkos::Experimental::details::execution_space
{

struct get_exec_t {
    template <typename Env>
        requires requires(const Env& env) { env.query(std::declval<get_exec_t>()); }
    [[nodiscard]] constexpr auto operator()(const Env& env) const noexcept {
        return env.query(*this);
    }

    template <typename Env>
        requires (!requires(const Env& env) { env.query(std::declval<get_exec_t>()); })
              && requires(const Env& env) { env.exec; }
    [[nodiscard]] constexpr auto operator()(const Env& env) const noexcept {
        return env.exec;
    }

    constexpr bool query(stdexec::forwarding_query_t) const noexcept {
        return true;
    }
};

inline constexpr get_exec_t get_exec{};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_QUERIES_HPP
