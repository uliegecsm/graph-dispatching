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

#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
#    include "plog/Log.h"
#endif

#include "kokkos_ext/impl/graph/bulk.hpp"
#include "kokkos_ext/impl/graph/continues_on.hpp"
#include "kokkos_ext/impl/graph/schedule_from.hpp"
#include "kokkos_ext/impl/graph/sync_wait.hpp"
#include "kokkos_ext/impl/graph/then.hpp"

namespace Kokkos::Experimental {

namespace details::graph {

struct Domain : public stdexec::default_domain {
    template <typename Tag, ::stdexec::sender Sndr, typename... Args>
    requires stdexec::__callable<apply_sender_for<Tag>, Sndr, Args...>
    static auto apply_sender(Tag, Sndr&& sndr, Args&&... args) {
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
        PLOG_DEBUG << Kokkos::Impl::TypeInfo<Domain>::name() << ": apply_sender for tag "
                   << Kokkos::Impl::TypeInfo<Tag>::name();
#endif
        return apply_sender_for<Tag>{}(std::forward<Sndr>(sndr), std::forward<Args>(args)...);
    }

    template <stdexec::sender Sndr, typename Env>
    requires stdexec::__callable<stdexec::__sexpr_apply_t, Sndr, transform_sender_for<stdexec::tag_of_t<Sndr>, Env>>
    static auto transform_sender(::stdexec::set_value_t, Sndr&& sndr, const Env& env_) {
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
        PLOG_DEBUG << Kokkos::Impl::TypeInfo<Domain>::name() << ": transform_sender for tag "
                   << Kokkos::Impl::TypeInfo<stdexec::tag_of_t<Sndr>>::name();
#endif
        return stdexec::__sexpr_apply(
            std::forward<Sndr>(sndr), transform_sender_for<stdexec::tag_of_t<Sndr>, Env>{.env_ = env_});
    }
};

//! The state is moveable, not copyable.
template <Kokkos::ExecutionSpace Exec>
struct State {
    using graph_t = Kokkos::Experimental::Graph<Exec>;

    explicit State(Exec exec_) // NOLINT(performance-unnecessary-value-param)
        : exec{std::move(exec_)} {
    }
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) = default;
    State& operator=(State&&) = default;
    ~State() = default;

    /**
     * @brief Build the graph lazily (the first time it is needed).
     *
     * @todo Make it a custom query.
     */
    auto& get_graph() {
        if (!graph) {
            graph.emplace(exec);
        }
        return *graph;
    }

    void submit() {
        if (graph.has_value()) {
            if (!is_submitted) {
                if (!is_instantiated) {
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
                    PLOG_DEBUG << "Instantiating the graph at " << std::addressof(*graph);
#endif
                    graph->instantiate();
                    is_instantiated = true;
                }
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
                PLOG_DEBUG << "Submitting the graph at " << std::addressof(*graph) << " on "
                           << Kokkos::Tools::Experimental::device_id(exec);
#endif
                Kokkos::Profiling::markEvent(std::format("{}: graph submit", Kokkos::Impl::TypeInfo<Exec>::name()));
                graph->submit(exec);
                is_submitted = true;
            }
        }
    }

    Exec exec;
    std::optional<graph_t> graph = std::nullopt;
    bool is_instantiated = false;
    bool is_submitted = false;
};

//! See https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/nvexec/stream/common.cuh#L168-L195).
template <Kokkos::ExecutionSpace Exec>
struct SchedulerEnv {
    [[nodiscard]]
    constexpr auto query(stdexec::get_completion_scheduler_t<stdexec::set_value_t>) const noexcept -> Scheduler<Exec> {
        return {state_ptr};
    }

    [[nodiscard]]
    constexpr auto query(stdexec::get_completion_domain_t<stdexec::set_value_t>) const noexcept -> Domain {
        return {};
    }

    State<Exec>* state_ptr = nullptr;
};

//! Scheduler for a @c Kokkos::Experimental::Graph.
template <Kokkos::ExecutionSpace Exec>
struct Scheduler {
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

        [[nodiscard]]
        constexpr auto get_env() const noexcept -> const SchedulerEnv<Exec>& {
            return env;
        }

        SchedulerEnv<Exec> env;
    };

    [[nodiscard]]
    auto schedule() const noexcept -> Sender {
        return {state_ptr};
    }

    [[nodiscard]]
    constexpr auto query(stdexec::get_completion_domain_t<::stdexec::set_value_t>) const noexcept -> Domain {
        return {};
    }

    [[nodiscard]]
    constexpr auto query(stdexec::get_completion_scheduler_t<stdexec::set_value_t>) const noexcept -> Scheduler {
        return {state_ptr};
    }

    friend bool operator==(const Scheduler&, const Scheduler&) noexcept = default;

    State<Exec>* state_ptr = nullptr;
};

} // namespace details::graph

//! Graph context using a @c Kokkos::Experimental::Graph under the hood.
template <Kokkos::ExecutionSpace Exec>
struct GraphContext {
    using state_t = details::graph::State<Exec>;

    state_t m_state;

    explicit GraphContext(Exec exec) // NOLINT(performance-unnecessary-value-param)
        : m_state{std::move(exec)} {
    }

    auto get_scheduler() const noexcept -> details::graph::Scheduler<Exec> {
        return {const_cast<state_t*>(&m_state)};
    }
};

} // namespace Kokkos::Experimental

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCONTEXT_HPP
