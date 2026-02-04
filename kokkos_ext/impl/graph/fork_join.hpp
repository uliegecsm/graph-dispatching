#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_FORK_JOIN_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_FORK_JOIN_HPP

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
#include <exec/fork_join.hpp>
PRAGMA_DIAGNOSTIC_POP

#include "kokkos_ext/impl/GraphContext_fwd.hpp"
#include "kokkos_ext/impl/env.hpp"
#include "kokkos_ext/impl/graph/Helpers.hpp"
#include "kokkos_ext/impl/graph/get_node.hpp"

namespace Kokkos::Experimental::details::graph {

/**
 * @brief Sender to be added to each of the @c exec::fork_join branch.
 *
 * It is inspired by https://github.com/NVIDIA/stdexec/blob/9ecb67154a739b79328d1a9b37c070cb7e4fc391/include/exec/fork_join.hpp#L90,
 * but is much simpler as the underlying @c Kokkos::Experimental::Graph does not allow
 * for a value channel.
 *
 * @todo Handle the error channel.
 */
template <typename Domain>
struct CacheSender {
    using sender_concept = stdexec::sender_t;

    template <stdexec::receiver Rcvr>
    struct CacheOpState {
        using operation_state_concept = stdexec::operation_state_t;

        void start() & noexcept {
            stdexec::set_value(std::move(rcvr));
        }

        Rcvr rcvr;
    };

    template <class Self, class... Env>
    static consteval auto get_completion_signatures() {
        return stdexec::completion_signatures<stdexec::set_value_t()>{};
    }

    template <stdexec::receiver Rcvr>
    constexpr auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) -> CacheOpState<Rcvr> {
        return {std::forward<Rcvr>(rcvr)};
    }

    static constexpr auto get_env() noexcept -> Kokkos::Experimental::details::impl::domain_queryable_env_t<Domain> {
        return {};
    }
};

//! Transform the closures to a proper @c stdexec::when_all.
struct make_when_all_fn {
    template <typename CacheSndr, typename... Closures>
    constexpr auto operator()(CacheSndr cache_sndr, Closures&&... closures) const {
        return stdexec::when_all(std::forward<Closures>(closures)(cache_sndr)...);
    }
};

template <typename Domain, typename PackedClosures>
using get_when_all_sndr_t = stdexec::__apply_result_t<make_when_all_fn, PackedClosures, CacheSender<Domain>>;

//! Operation state for @c exec::fork_join.
template <stdexec::scheduler Schd, typename Sndr, typename PackedClosures, typename InnerRcvr>
struct ForkJoinOpState {
    using operation_state_concept = stdexec::operation_state_t;

    template <typename NodeType>
    struct ForkJoinReceiver {
        using receiver_concept = stdexec::receiver_t;

        ForkJoinOpState* opstate;
        stdexec::__rcvr_ref_t<InnerRcvr> inner_rcvr;

        void set_value() && noexcept {
            stdexec::set_value(std::move(inner_rcvr));
        }

        template <typename Error>
        void set_error(Error&& error) && noexcept {
            stdexec::set_error(std::move(inner_rcvr), std::forward<Error>(error));
        }

        GRAPH_DISPATCHING_KOKKOS_EXT_JOIN_NODE(InnerRcvr, inner_rcvr, NodeType, opstate->fork_node)
    };

    using env_t = stdexec::__fwd_env_t<stdexec::env_of_t<InnerRcvr>>;
    using domain_t = stdexec::__completion_domain_of_t<stdexec::set_value_t, Sndr, env_t>;
    using fork_completions_t = stdexec::completion_signatures_of_t<Sndr, env_t>;
    using when_all_sndr_t = get_when_all_sndr_t<domain_t, PackedClosures>;
    using fork_opstate_t = stdexec::connect_result_t<Sndr, stdexec::__rcvr_ref_t<ForkJoinOpState, env_t>>;
    using fork_node_t = std::remove_cvref_t<get_predecessor_t<fork_opstate_t, env_t, typename Schd::execution_space>>;
    using join_opstate_t = stdexec::connect_result_t<when_all_sndr_t, ForkJoinReceiver<fork_node_t>>;
    using cache_sndr_t = CacheSender<domain_t>;

    Schd schd;
    InnerRcvr inner_rcvr;
    fork_opstate_t fork_opstate;
    fork_node_t fork_node;
    join_opstate_t join_opstate;

    ForkJoinOpState(Schd&& schd_, Sndr&& sndr, PackedClosures&& packed_closures, InnerRcvr&& inner_rcvr_)
        : schd(std::move(schd_))
        , inner_rcvr(std::move(inner_rcvr_))
        , fork_opstate(stdexec::connect(std::move(sndr), stdexec::__ref_rcvr(*this)))
        , fork_node(get_predecessor(fork_opstate, this->get_env(), schd.state_ptr->get_graph()))
        , join_opstate(
              stdexec::connect(
                  stdexec::__apply(make_when_all_fn{}, std::move(packed_closures), cache_sndr_t{}),
                  ForkJoinReceiver<fork_node_t>{this, stdexec::__ref_rcvr(inner_rcvr)})) {
    }

    decltype(auto) query(get_node_t) const noexcept {
        return join_opstate.query(get_node);
    }

    void start() & noexcept {
        stdexec::start(fork_opstate);
    }

    //! @todo Properly deal with the error channel by setting the error value in the cache sender.
    template <typename Tag, typename... Args>
    constexpr void _complete(Tag, Args&&...) noexcept {
        stdexec::start(join_opstate);
    }

    constexpr void set_value() noexcept {
        this->_complete(stdexec::set_value);
    }

    template <class Error>
    constexpr void set_error(Error&& error) noexcept {
        this->_complete(stdexec::set_error, std::forward<Error>(error));
    }

    GRAPH_DISPATCHING_KOKKOS_EXT_FORWARDING_GET_ENV(InnerRcvr, inner_rcvr)
};

//! Sender for @c exec::fork_join.
template <stdexec::scheduler Schd, typename Sndr, typename PackedClosures>
struct ForkJoinSender {
    using sender_concept = stdexec::sender_t;

    template <class Self, class... Env>
    static consteval auto get_completion_signatures() {
        using sndr_t = stdexec::__copy_cvref_t<Self, Sndr>;
        using sndr_completions_t = stdexec::completion_signatures_of_t<sndr_t, stdexec::__fwd_env_t<Env>...>;
        using when_all_sndr_t = get_when_all_sndr_t<sndr_completions_t, PackedClosures>;
        return stdexec::completion_signatures_of_t<when_all_sndr_t, stdexec::__fwd_env_t<Env>...>{};
    }

    template <::stdexec::receiver Rcvr>
    ::stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
        return ForkJoinOpState<Schd, Sndr, PackedClosures, std::remove_cvref_t<Rcvr>>(
            std::move(schd), std::move(sndr), std::move(packed_closures), std::forward<Rcvr>(rcvr));
    }

    GRAPH_DISPATCHING_KOKKOS_EXT_FORWARDING_GET_ENV(Sndr, sndr)

    Schd schd;
    Sndr sndr;
    PackedClosures packed_closures;
};

template <typename Env>
struct transform_sender_for<exec::fork_join_t, Env> {
    template <typename PackedClosures, stdexec::sender Sndr>
    auto operator()(exec::fork_join_t, PackedClosures&& closures, Sndr&& sndr) && noexcept {
        static_assert(stdexec::__is_instance_of<PackedClosures, stdexec::__tuple>);

        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env_);
        static_assert(stdexec::__is_instance_of<decltype(schd), Scheduler>);

        return ForkJoinSender<decltype(schd), Sndr, PackedClosures>{
            .schd = std::move(schd),
            .sndr = std::forward<Sndr>(sndr),
            .packed_closures = std::forward<PackedClosures>(closures)};
    }

    const Env& env_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

} // namespace Kokkos::Experimental::details::graph

// NOLINTBEGIN(bugprone-reserved-identifier)
namespace stdexec::__detail {
template <stdexec::scheduler Schd, typename Sndr, typename PackedClosures, typename InnerRcvr>
extern __declfn_t<Kokkos::Experimental::details::graph::ForkJoinOpState<
    Schd,
    __demangle_t<Sndr>,
    PackedClosures,
    __demangle_t<InnerRcvr>
>>
    __demangle_v<Kokkos::Experimental::details::graph::ForkJoinOpState<Schd, Sndr, PackedClosures, InnerRcvr>>;
} // namespace stdexec::__detail
// NOLINTEND(bugprone-reserved-identifier)

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_FORK_JOIN_HPP
