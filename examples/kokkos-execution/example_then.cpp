#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"

#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"

#include "kokkos_ext/impl/GraphContext.hpp"

#include "tests/Functors.hpp"

/**
 * @addtogroup examples
 *
 * A @c then node with @c Kokkos
 * -----------------------------
 *
 * Create a graph with @c then nodes either using @c Kokkos directly
 * or *via* @c stdexec customization.
 *
 * The examples can be found in @ref examples/kokkos-execution/example_then.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace examples::KokkosExecution::then {

struct GraphThenTest
    : public ::testing::Test
    , public Kokkos::utils::tests::scoped::ExecutionSpace<execution_space> {
   public:
    using view_s_t = Kokkos::View<int, Kokkos::SharedSpace>;
    using functor_t = tests::ThenFunctor<view_s_t>;

   public:
    void SetUp() override {
        this->data = view_s_t(Kokkos::view_alloc(exec, "data"));
    }

   protected:
    view_s_t data;
};

//! @test Vanilla @c Kokkos.
TEST_F(GraphThenTest, kokkos_vanilla) {
    auto graph = Kokkos::Experimental::create_graph(exec, [&](const auto& root) {
        root.then("node A", functor_t{.data = data})
            .then("node B", functor_t{.data = data})
            .then("node C", functor_t{.data = data});
    });

    graph.submit(exec);
    exec.fence();

    ASSERT_EQ(data(), 3);
}

//! @test Using @c stdexec customization.
TEST_F(GraphThenTest, kokkos_execution) {
    const Kokkos::Experimental::GraphContext<execution_space> gctx{exec};

    stdexec::sender auto chain = stdexec::schedule(gctx.get_scheduler()) | stdexec::then(functor_t{.data = data})
                               | stdexec::then(functor_t{.data = data}) | stdexec::then(functor_t{.data = data});

    stdexec::sync_wait(std::move(chain));

    ASSERT_EQ(data(), 3);
}

} // namespace examples::KokkosExecution::then
