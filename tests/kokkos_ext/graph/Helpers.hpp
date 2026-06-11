#ifndef GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_GRAPH_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_GRAPH_HELPERS_HPP

#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
#    include "plog/Formatters/TxtFormatter.h"
#    include "plog/Initializers/ConsoleInitializer.h"
#    include "plog/Log.h"
#endif

#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"

#include "kokkos_ext/impl/GraphContext.hpp"

#include "tests/Functors.hpp"
#include "tests/utils/Graph.hpp"

namespace tests::kokkos_ext::impl {
template <typename Exec>
struct GraphContextTest
    : public virtual ::testing::Test
    , public Kokkos::utils::tests::scoped::ExecutionSpace<Exec> {
   public:
    using context_t = Kokkos::Experimental::GraphContext<Exec>;
    using scheduler_t = decltype(std::declval<const context_t&>().get_scheduler());
    using schedule_sender_t = decltype(::stdexec::schedule(std::declval<scheduler_t>()));

    using value_t = int;
    using view_s_t = Kokkos::View<value_t, Kokkos::SharedSpace>;

   public:
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
    static void SetUpTestSuite() {
        ::plog::init<::plog::TxtFormatter>(::plog::debug, ::plog::streamStdOut);
    }
#endif
};

//! Check the node type.
template <typename ObjType, typename NodeType>
constexpr bool check_node_type() {
    static_assert(std::same_as<
                  ::stdexec::__query_result_t<const ObjType&, Kokkos::Experimental::details::graph::get_node_t>,
                  NodeType
    >);
    return true;
}

} // namespace tests::kokkos_ext::impl

#endif // GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_GRAPH_HELPERS_HPP
