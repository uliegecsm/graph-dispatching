#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_SYNCWAIT_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_SYNCWAIT_HPP

#include "Kokkos_Core_fwd.hpp"

namespace Kokkos::Experimental::graph::details
{
template <typename Exec> requires Kokkos::is_execution_space_v<Exec>
struct SyncWaitReceiver
{
    Exec exec;

    void set_value() && { exec.fence(); }
};

//! @todo Short description.
struct SyncWait
{
    template <typename Sender> requires sender<std::remove_cvref_t<Sender>>
    void operator()(Sender&& sndr) const
    {
        std::forward<Sender>(sndr).connect(
            SyncWaitReceiver{sndr.get_env().exec}
        ).start();
    }
};

struct ReadyReceiver
{
    std::string label = "ReadyReceiver";

    void set_value() && { /* nothing to do */ }
};

//! This is for underlying graph to be ready.
struct Ready
{
    template <typename Graph>
    decltype(auto) operator()(Graph&& g) const
    {
        auto graph = g.get_env().graph;
        std::forward<Graph>(g).connect(
            ReadyReceiver{}
        ).start();
        return graph;
    }
};

} // namespace Kokkos::Experimental::graph::details

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_SYNCWAIT_HPP
