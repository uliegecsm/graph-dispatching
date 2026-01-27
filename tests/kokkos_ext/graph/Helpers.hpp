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

namespace tests::kokkos_ext::impl {
template <typename Exec>
struct GraphContextTest
    : public virtual ::testing::Test
    , public Kokkos::utils::tests::scoped::ExecutionSpace<Exec> {
   public:
    using context_t = Kokkos::Experimental::GraphContext<Exec>;
    using scheduler_t = decltype(std::declval<const context_t&>().get_scheduler());
    using schedule_sender_t = decltype(::stdexec::schedule(std::declval<scheduler_t>()));

    using view_s_t = Kokkos::View<int, Kokkos::SharedSpace>;
    using view_sa_t = Kokkos::View<int, Kokkos::SharedSpace, Kokkos::MemoryTraits<Kokkos::Atomic>>;

   public:
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
    static void SetUpTestSuite() {
        ::plog::init<::plog::TxtFormatter>(::plog::debug, ::plog::streamStdOut);
    }
#endif
};

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#    define KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(_exec_)
#else
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#    define KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(_exec_)                                                                \
        MATCHER_FOR_BEGIN_FENCE(_exec_, "Kokkos::DefaultGraph::submit: fencing before launching graph nodes"),
#endif

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(_exec_predecessor_)                                                           \
    MATCHER_FOR_BEGIN_FENCE(_exec_predecessor_, "Kokkos::DefaultGraphNode::execute_node: sync with predecessors")

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_DEFAULTED_GRAPH_ENDOF_SINK_SYNC(_exec_)                                                                 \
    MATCHER_FOR_BEGIN_FENCE(_exec_, "Kokkos::DefaultGraph::submit: fencing before ending graph submit")

//! Inspired by https://github.com/kokkos/kokkos/blob/02eba5e5a94173a6d580638eb92a7357e2f9a7f8/core/unit_test/TestGraph.hpp#L1142-L1158.
template <Kokkos::ExecutionSpace>
struct GraphIsDefaulted : std::true_type { };

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)                                                          \
    || (defined(KOKKOS_ENABLE_SYCL) && defined(KOKKOS_IMPL_SYCL_GRAPH_SUPPORT))
template <>
struct GraphIsDefaulted<Kokkos::DefaultExecutionSpace> : std::false_type { };
#endif

template <Kokkos::ExecutionSpace Exec>
constexpr bool is_graph_defaulted = GraphIsDefaulted<Exec>::value;

} // namespace tests::kokkos_ext::impl

#endif // GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_GRAPH_HELPERS_HPP
