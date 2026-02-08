#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_WHEN_ALL_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_WHEN_ALL_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/GraphContext_fwd.hpp"
#include "kokkos_ext/impl/completion_signatures.hpp"

namespace Kokkos::Experimental::details::graph {

/**
 * @brief Operation state for a @c when_all whose children are all completing on @ref Kokkos::Experimental::details::graph::Domain.
 *
 * @note It is assumed they all belong to the same underlying graph.
 */
template <stdexec::receiver InnerRcvr, ::stdexec::sender... Sndrs>
struct WhenAllOpState {
    using operation_state_concept = stdexec::operation_state_t;

    //! Receiver for a child of @c when_all.
    struct WhenAllChildReceiver {
        using receiver_concept = stdexec::receiver_t;

        WhenAllOpState* parent;

        void set_value() noexcept {
            parent->child_completed();
        }

        template <class Error>
        void set_error(Error&& err) noexcept {
            parent->child_error(std::forward<Error>(err));
        }

        [[nodiscard]]
        constexpr auto get_env() const noexcept -> stdexec::env_of_t<InnerRcvr> {
            return parent->get_env();
        }
    };

    using children_opstates_t = stdexec::__tuple<stdexec::connect_result_t<Sndrs, WhenAllChildReceiver>...>;

    InnerRcvr inner_rcvr;
    children_opstates_t children_opstates;
    std::exception_ptr error = nullptr;

    using node_t = decltype(stdexec::__apply(
        [](auto&... ops) { return Kokkos::Experimental::when_all(*ops.get_node()...); },
        std::declval<children_opstates_t&>()));

    std::optional<node_t> node = std::nullopt;

    WhenAllOpState(stdexec::__tuple<Sndrs...>&& sndrs_, InnerRcvr&& inner_rcvr_)
        : inner_rcvr(std::move(inner_rcvr_))
        , children_opstates(
              stdexec::__apply(
                  [this]<typename... Children>(Children&&... children) -> children_opstates_t {
                      return children_opstates_t{
                          stdexec::connect(std::forward<Children>(children), WhenAllChildReceiver{this})...};
                  },
                  std::move(sndrs_)))
        , node(this->create_node()) {
    }

    WhenAllOpState(const WhenAllOpState&) = delete;
    WhenAllOpState& operator=(const WhenAllOpState&) = delete;
    WhenAllOpState(WhenAllOpState&&) = delete;
    WhenAllOpState& operator=(WhenAllOpState&&) = delete;
    ~WhenAllOpState() = default;

    auto create_node() {
        return stdexec::__apply(
            [](auto&... ops) { return Kokkos::Experimental::when_all(*ops.get_node()...); }, children_opstates);
    }

    decltype(auto) get_node() const {
        return node;
    }

    //! Children will start and most probably complete very quickly, before the corresponding graph nodes are done.
    void start() & noexcept {
        stdexec::__apply([](auto&... ops) -> void { (stdexec::start(ops), ...); }, children_opstates);

        //! @todo Find a better way to do this conditionally depending on what's next.
        constexpr bool skip = stdexec::__queryable_with<stdexec::env_of_t<InnerRcvr>, execution_space::get_exec_t>;
        if (!skip) {
            stdexec::__get<0>(children_opstates)
                .schd.state_ptr->wait(
                    std::format(
                        "{}: when_all",
                        Kokkos::Impl::TypeInfo<decltype(stdexec::__get<0>(children_opstates)
                                                            .schd.state_ptr->exec)>::name()));
        }

        if (error != nullptr) {
            stdexec::set_error(std::move(inner_rcvr), std::move(error));
        } else {
            stdexec::set_value(std::move(inner_rcvr));
        }
    }

    void child_completed() noexcept {
    }

    template <class Error>
    void child_error(Error&& err) noexcept {
        this->error = std::forward<Error>(err);
    }

    //! @todo Check if this is correct.
    auto get_env() const noexcept -> stdexec::env_of_t<InnerRcvr> {
        return stdexec::get_env(inner_rcvr);
    }
};

/**
 * @brief Sender for @c stdexec::when_all.
 *
 * @warning It only accepts input senders that complete on the @ref Kokkos::Experimental::details::graph::Domain domain,
 *          because it adds a @c Kokkos::Experimental::when_all node to the underlying graph.
 */
template <::stdexec::sender... Sndrs>
struct WhenAllSender {
    using sender_concept = ::stdexec::sender_t;

    struct attrs {
        template <typename... Env>
        [[nodiscard]]
        constexpr auto
            query(stdexec::get_completion_domain_t<stdexec::set_value_t>, const Env&...) const noexcept -> Domain {
            return {};
        }
    };

    //! @todo Make it much more robust as per https://github.com/NVIDIA/stdexec/blob/0f8338ab55b03d63f5fef111cefed7ab4002e78d/include/stdexec/__detail/__when_all.hpp#L341.
    template <typename Self, typename... Env>
    static consteval auto get_completion_signatures() {
        return stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>{};
    }

    template <::stdexec::receiver Rcvr>
    ::stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
        return WhenAllOpState<Rcvr, Sndrs...>(std::move(sndrs), std::forward<Rcvr>(rcvr));
    }

    constexpr auto get_env() const noexcept -> attrs {
        return {};
    }

    stdexec::__tuple<Sndrs...> sndrs;
};

template <typename Env>
struct transform_sender_for<stdexec::when_all_t, Env> {
    template <::stdexec::sender... Sndrs>
    requires(graph_completing_sender<Sndrs, Env> && ...)
    auto operator()(stdexec::when_all_t, ::stdexec::__ignore, Sndrs&&... sndrs) && noexcept {
        return WhenAllSender<Sndrs...>{.sndrs = {std::forward<Sndrs>(sndrs)...}};
    }

    const Env& env_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

} // namespace Kokkos::Experimental::details::graph

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_WHEN_ALL_HPP
