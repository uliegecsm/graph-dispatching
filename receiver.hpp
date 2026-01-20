#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_RECEIVER_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_RECEIVER_HPP

#include "stdexec/execution.hpp"

namespace Kokkos::Experimental::details::execution_space {
struct ReceiverBase {
    using receiver_concept = stdexec::receiver_t;
};

template <stdexec::scheduler Schd, stdexec::receiver Rcvr>
struct Receiver : public ReceiverBase {
    template <typename Scheduler, typename Rec>
    Receiver(Scheduler&& scheduler, Rec&& receiver)
        : schd(std::forward<Scheduler>(scheduler))
        , rcvr(std::forward<Rec>(receiver)) {
    }

    Schd schd;
    Rcvr rcvr;
};

template <typename Rcvr>
concept receiver = stdexec::receiver<Rcvr> && std::is_base_of_v<ReceiverBase, Rcvr>;

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_RECEIVER_HPP
