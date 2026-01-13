#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCONTEXT_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCONTEXT_HPP

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
#include <stdexec/execution.hpp>
PRAGMA_DIAGNOSTIC_POP

#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"

#include "kokkos_ext/impl/GraphContext_fwd.hpp"

namespace Kokkos::Experimental {

namespace details::graph {

struct Domain : public stdexec::default_domain { };

//! See https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/nvexec/stream/common.cuh#L168-L195).
template <typename Exec>
requires Kokkos::is_execution_space_v<Exec>
struct GraphSchedulerEnv {
    [[nodiscard]]
    constexpr auto
        query(stdexec::get_completion_scheduler_t<stdexec::set_value_t>) const noexcept -> GraphScheduler<Exec> {
        return {graph_ptr};
    }

    [[nodiscard]]
    constexpr auto query(stdexec::get_completion_domain_t<stdexec::set_value_t>) const noexcept -> Domain {
        return {};
    }

    bool operator==(const GraphSchedulerEnv&) const noexcept = default;

    Kokkos::Experimental::Graph<Exec>* graph_ptr = nullptr;
};

//! Scheduler for a @c Kokkos::Experimental::Graph.
template <typename Exec>
requires Kokkos::is_execution_space_v<Exec>
struct GraphScheduler {
    //! As per https://eel.is/c++draft/exec.sched#1.
    using scheduler_concept = stdexec::scheduler_t;

    template <stdexec::receiver Rcvr>
    struct OpState {
        using operation_state_concept = stdexec::operation_state_t;

        Rcvr rcvr;

        //! @todo Check signature. And check whether we should move the receiver.
        void start() & noexcept {
            stdexec::set_value(std::move(rcvr));
        }
    };

    struct Sender {
        using sender_concept = stdexec::sender_t;

        using completion_signatures = ::stdexec::completion_signatures<::stdexec::set_value_t()>;

        template <stdexec::receiver_of<completion_signatures> Rcvr>
        OpState<std::remove_cvref_t<Rcvr>>
            connect(Rcvr&& rcvr) noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<Rcvr>, Rcvr&&>) {
            return {std::forward<Rcvr>(rcvr)};
        }

        auto& get_env() const noexcept {
            return env;
        }

        GraphSchedulerEnv<Exec> env;
    };

    auto schedule() const noexcept -> Sender {
        return {.env = env};
    }

    [[nodiscard]]
    constexpr auto query(stdexec::get_completion_domain_t<::stdexec::set_value_t>) const noexcept -> Domain {
        return {};
    }

    [[nodiscard]]
    constexpr auto query(stdexec::get_completion_scheduler_t<stdexec::set_value_t>) const noexcept -> GraphScheduler {
        return GraphScheduler{env};
    }

    bool operator==(const GraphScheduler&) const noexcept = default;

    GraphSchedulerEnv<Exec> env;
};

//! Deduction guide for @ref GraphScheduler.
template <typename Exec>
GraphScheduler(Exec&&) -> GraphScheduler<std::remove_cvref_t<Exec>>;

} // namespace details::graph

//! Graph context using a @c Kokkos::Experimental::Graph under the hood.
template <typename Exec>
requires Kokkos::is_execution_space_v<Exec>
struct GraphContext {
    Kokkos::Experimental::Graph<Exec> graph;

    explicit GraphContext(const Exec& exec)
        : graph{exec} { };

    auto get_scheduler() const noexcept -> details::graph::GraphScheduler<Exec> {
        return {&graph};
    }
};

} // namespace Kokkos::Experimental

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCONTEXT_HPP
