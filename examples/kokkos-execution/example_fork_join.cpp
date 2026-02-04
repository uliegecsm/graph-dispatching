#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"

#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"

#include "kokkos_ext/impl/GraphContext.hpp"

#include "examples/kokkos-execution/diamond.hpp"
#include "tests/Functors.hpp"

/**
 * @addtogroup examples
 *
 * A fork/join graph with @c Kokkos
 * --------------------------------
 *
 * Create a graph with a fork/join topology (*a.k.a.* diamond)
 * either using @c Kokkos directly
 * or *via* @c stdexec customization.
 *
 * The examples can be found in @ref examples/kokkos-execution/example_fork_join.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace examples::KokkosExecution::fork_join {

struct GraphForkJoinTest
    : public ::testing::Test
    , public Kokkos::utils::tests::scoped::ExecutionSpace<execution_space> {
   public:
    using memory_space = typename execution_space::memory_space;

    static constexpr size_t size = 10;
    static_assert(size % 2 == 0);

    using view_t = Kokkos::View<int[size], memory_space>;
    using functor_t = tests::AddValueOffset<view_t>;

   public:
    void SetUp() override {
        this->data = view_t(Kokkos::view_alloc(exec, "data"));
    }

   protected:
    view_t data;
};

//! @test Vanilla @c Kokkos.
TEST_F(GraphForkJoinTest, kokkos_vanilla) {
    auto graph = Kokkos::Experimental::create_graph(exec, [&](const auto& root) {
        const auto node_A =
            root.then_parallel_for("node A", size, functor_t{.data = data, .value = diamond::Values::value_A});

        const auto node_B = node_A.then_parallel_for(
            "node B", size / 2, functor_t{.data = data, .value = diamond::Values::value_B, .offset = 0});
        const auto node_C = node_A.then_parallel_for(
            "node C", size / 2, functor_t{.data = data, .value = diamond::Values::value_C, .offset = size / 2});

        const auto node_D =
            Kokkos::Experimental::when_all(node_B, node_C)
                .then_parallel_for("node D", size, functor_t{.data = data, .value = diamond::Values::value_D});
    });

    graph.submit(exec);
    exec.fence();

    ASSERT_TRUE(diamond::Values::check(exec, data));
}

#if HAS_FORK_JOIN_CUSTOMIZED
//! @test Using @c stdexec customization.
TEST_F(GraphForkJoinTest, kokkos_execution) {
    const Kokkos::Experimental::GraphContext<execution_space> gctx{exec};

    auto node_A =
        stdexec::schedule(gctx.get_scheduler())
        | stdexec::bulk(stdexec::par, size, functor_t{.data = data, .value = diamond::Values::value_A})
        | exec::fork_join(
            stdexec::continues_on(gctx.get_scheduler())
                | stdexec::bulk(
                    stdexec::par, size / 2, functor_t{.data = data, .value = diamond::Values::value_B, .offset = 0}),
            stdexec::continues_on(gctx.get_scheduler())
                | stdexec::bulk(
                    stdexec::par,
                    size / 2,
                    functor_t{.data = data, .value = diamond::Values::value_C, .offset = size / 2}))
        | stdexec::bulk(stdexec::par, size, functor_t{.data = data, .value = diamond::Values::value_D});

    stdexec::sync_wait(std::move(node_D));

    ASSERT_TRUE(diamond::Values::check(exec, data));
}
#endif
} // namespace examples::KokkosExecution::fork_join
