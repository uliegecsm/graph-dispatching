#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_PARALLELREDUCE_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_PARALLELREDUCE_HPP

#include "kokkos_ext/impl/PartialAlgorithm.hpp"
#include "kokkos_ext/impl/Utils.hpp"

namespace Kokkos::Experimental::graph::details
{

//! Specialization for @c Kokkos parallel-reduce.
template <typename Policy, typename Functor, typename Reducer>
struct PartialAlgorithm<Kokkos::ParallelReduceTag, std::string, Policy, Functor, Reducer>
{
    std::string label;
    Policy policy;
    Functor functor;
    Reducer reducer;

    template <typename Sender> requires is_graph_sender<std::remove_reference_t<Sender>>
    decltype(auto) operator()(Sender&& input) &&
    {
        return std::forward<Sender>(input).then_parallel_reduce(
            std::move(label),
            std::move(policy),
            std::move(functor),
            std::move(reducer)
        );
    }

    /// Case of input sender being a @c Kokkos execution space instance.
    /// We do eager execution, to get as close as possible from what a user currently
    /// expects with vanilla @c Kokkos code.
    template <typename Sender> requires (! is_graph_sender<std::remove_reference_t<Sender>>)
    decltype(auto) operator()(Sender&& exec) &&
    {
        static_assert(false, "merde");
        Kokkos::parallel_reduce(
            std::move(label),
            details::update_policy(std::move(policy), exec),
            std::move(functor),
            std::move(reducer)
        );
        return std::forward<Sender>(exec);
    }
};

} // namespace Kokkos::Experimental::graph::details

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_PARALLELREDUCE_HPP
