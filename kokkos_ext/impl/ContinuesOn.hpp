#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_CONTINUESON_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_CONTINUESON_HPP

#include "kokkos_ext/impl/Concepts.hpp"

namespace Kokkos::Experimental::details
{
//! Helper for @ref Kokkos::Experimental::continues_on.
template <typename Exec>
struct ContinuesOn
{
    Exec exec;

    /**
     * Check if the execution context @ref exec is the same as @p sender, otherwise fence.
     *
     * Return @ref exec.
     *
     * @note For now, we allow @ref exec and @p sender to point to the same execution context.
     */
    template <typename Sender> requires (! graph::details::is_graph_sender<std::remove_reference_t<Sender>>)
    decltype(auto) operator()(Sender&& sender) &&
    {
        if(exec != sender) sender.fence();
        return std::move(exec);
    }
};

} // namespace Kokkos::Experimental::details

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_CONTINUESON_HPP
