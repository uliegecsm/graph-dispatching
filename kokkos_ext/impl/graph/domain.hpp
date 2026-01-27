#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_DOMAIN_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_DOMAIN_HPP

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
#include <stdexec/execution.hpp>
PRAGMA_DIAGNOSTIC_POP

#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
#    include "plog/Log.h"
#endif

#include "kokkos_ext/impl/GraphContext_fwd.hpp"

namespace Kokkos::Experimental::details::graph {

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
    requires stdexec::__applicable<transform_sender_for<stdexec::tag_of_t<Sndr>, Env>, Sndr>
    static auto transform_sender(::stdexec::set_value_t, Sndr&& sndr, const Env& env_) {
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
        PLOG_DEBUG << Kokkos::Impl::TypeInfo<Domain>::name() << ": transform_sender for tag "
                   << Kokkos::Impl::TypeInfo<stdexec::tag_of_t<Sndr>>::name();
#endif
        return stdexec::__apply(
            transform_sender_for<stdexec::tag_of_t<Sndr>, Env>{.env_ = env_}, std::forward<Sndr>(sndr));
    }
};

} // namespace Kokkos::Experimental::details::graph

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_DOMAIN_HPP
