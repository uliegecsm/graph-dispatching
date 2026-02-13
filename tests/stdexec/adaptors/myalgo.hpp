#ifndef GRAPH_DISPATCHING_TESTS_STDEXEC_ADAPTORS_MYALGO_HPP
#define GRAPH_DISPATCHING_TESTS_STDEXEC_ADAPTORS_MYALGO_HPP

#include "stdexec/execution.hpp"

/**
 * This file shows how to define a new, customizable algorithm.
 *
 * The algorithm allows any type of value to be taken, and the goal is to dump it to @c std::cout.
 * A customization is allowed to add contextual/debug information in the dump.
 * Any value from the value channel is forwarded.
 *
 * It is inspired by:
 *  - https://github.com/NVIDIA/stdexec/blob/514f239f7dcde14a73fc5013a46d8ec77a1f7e2f/include/stdexec/__detail/__then.hpp
 */

namespace myalgo_namespace {
namespace details {

//! The tag for the new algorithm.
struct myalgo_t {
    //! Allows the @c stdexec::transform_sender machinery (sender adaptor).
    template <stdexec::sender Sndr, typename Obj>
    constexpr auto operator()(Sndr&& sndr, Obj&& obj) const -> stdexec::__well_formed_sender auto {
        return stdexec::__make_sexpr<myalgo_t>(std::forward<Obj>(obj), std::forward<Sndr>(sndr));
    }

    template <typename Obj>
    constexpr auto operator()(Obj&& obj) const {
        return stdexec::__closure(*this, std::forward<Obj>(obj));
    }
};

//! Default implementation.
struct myalgo_impl : stdexec::__sexpr_defaults {
    template <typename Sndr, typename... Env>
    using _completion_signatures_t = stdexec::completion_signatures_of_t<Sndr, Env...>;

    template <typename Sndr, typename... Env>
    static consteval auto __get_completion_signatures() // NOLINT(bugprone-reserved-identifier)
        -> _completion_signatures_t<stdexec::__child_of<Sndr>, Env...> {
        static_assert(stdexec::sender_expr_for<Sndr, myalgo_t>);
        return {};
    };

    struct _complete_fn {
        template <typename IndexType, typename Tag, typename State, typename... Args>
        constexpr void operator()(IndexType, State& state, Tag, Args&&... args) const noexcept {
            static_assert(std::same_as<IndexType, std::integral_constant<unsigned long, 0>>);
            static_assert(stdexec::__is_instance_of<State, stdexec::__detail::__state>);

            if constexpr (std::same_as<Tag, stdexec::set_value_t>) {
                std::cout << "myalgo_t: " << static_cast<State&&>(state).__data_ << '\n';
                stdexec::set_value(static_cast<State&&>(state).__rcvr_, std::forward<Args>(args)...);
            } else {
                Tag()(static_cast<State&&>(state).__rcvr_, std::forward<Args>(args)...);
            }
        }
    };

    static constexpr auto __complete = _complete_fn{}; // NOLINT(bugprone-reserved-identifier)
};
} // namespace details

using details::myalgo_t;

inline constexpr myalgo_t myalgo{};

} // namespace myalgo_namespace

namespace stdexec {
template <>
struct __sexpr_impl<myalgo_namespace::myalgo_t> : myalgo_namespace::details::myalgo_impl { };
} // namespace stdexec

#endif // GRAPH_DISPATCHING_TESTS_STDEXEC_ADAPTORS_MYALGO_HPP
