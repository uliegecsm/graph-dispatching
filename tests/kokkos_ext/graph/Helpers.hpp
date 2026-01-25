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

} // namespace tests::kokkos_ext::impl

#endif // GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_GRAPH_HELPERS_HPP
