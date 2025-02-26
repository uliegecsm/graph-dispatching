#ifndef GRAPH_DISPATCHING_TESTS_STDEXEC_UTILS_HPP
#define GRAPH_DISPATCHING_TESTS_STDEXEC_UTILS_HPP

#include "stdexec/execution.hpp"

namespace tests::stdexec
{

template <class Sndr, class... Signatures>
concept has_completion_signatures = std::same_as<
    std::invoke_result_t<::stdexec::get_completion_signatures_t, Sndr>,
    ::stdexec::completion_signatures<Signatures...>
>;

} // namespace tests::stdexec

#endif // GRAPH_DISPATCHING_TESTS_STDEXEC_UTILS_HPP
