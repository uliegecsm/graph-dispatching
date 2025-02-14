#ifndef GRAPH_DISPATCHING_TESTS_STDEXEC_UTILS_HPP
#define GRAPH_DISPATCHING_TESTS_STDEXEC_UTILS_HPP

#include "stdexec/execution.hpp"

namespace tests::stdexec
{

//! Adapted from https://github.com/NVIDIA/stdexec/blob/b888185d667f68b9a8bda5d0c81d03edf9ec3fe1/include/stdexec/__detail/__execution_fwd.hpp#L124-L126.
template <class Sndr, class CPO>
concept has_completion_scheduler = ::stdexec::tag_invocable<
    ::stdexec::get_completion_scheduler_t<CPO>,
    ::stdexec::env_of_t<const Sndr&>
>;

template <class Sndr, class... Signatures>
concept has_completion_signatures = std::same_as<
    std::invoke_result_t<::stdexec::get_completion_signatures_t, Sndr>,
    ::stdexec::completion_signatures<Signatures...>
>;

} // namespace tests::stdexec

#endif // GRAPH_DISPATCHING_TESTS_STDEXEC_UTILS_HPP
