#include "benchmark/benchmark.h"

#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"

#include "kokkos-utils/timer/Timer.hpp"

#include "benchmarks/Helpers.hpp"

#if defined(KOKKOS_ENABLE_SYCL) && !defined(KOKKOS_IMPL_SYCL_GRAPH_SUPPORT)
    #error "KOKKOS_IMPL_SYCL_GRAPH_SUPPORT is required."
#endif

/**
 * @addtogroup unitbenchmarks
 *
 * Fork graph benchmark
 * --------------------
 *
 * Compare @c Kokkos::Experimental::Graph with an execution-space-instance-pool-based implementation for a
 * fork graph.
 *
 * The benchmarks can be found in @ref benchmarks/graph/benchmark_fork.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace benchmarks::graph
{

template <typename ViewType, typename IndexType>
struct Increment {
    ViewType data;
    IndexType index;

    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T) const noexcept {
        ++data(index);
    }
};

class ForkBenchmark : public benchmarks::BenchmarkBase
{
protected:
    using graph_t = Kokkos::Experimental::Graph<execution_space>;
    using view_t = Kokkos::View<unsigned int*, execution_space>;
    using state_range_t = int64_t;

    static constexpr unsigned short int index_num_branches = 0;
    static constexpr unsigned short int index_num_submits = 1;
    static constexpr unsigned short int index_num_nodes = 2;

public:
    //! We need to create @c Kokkos objects in the @ref SetUp, not using the constructor or in-class default member initializers.
    void SetUp(const ::benchmark::State& state) override {
        const std::vector<state_range_t> weights(state.range(index_num_branches), 1);
        this->execs = Kokkos::Experimental::partition_space(
            execution_space{},
            weights
        );
        this->data = view_t(Kokkos::view_alloc(execs.at(0)), state.range(index_num_branches));
    }

    void TearDown(const ::benchmark::State&) override {
        this->data = view_t();
        this->execs.clear();
    }

    FIXME_PARTIAL_OVERRIDE_WARNING_CUDA(SetUp)
    FIXME_PARTIAL_OVERRIDE_WARNING_CUDA(TearDown)

    void run_exec(::benchmark::State& state) const
    {
        for (auto sample : state) {
            for(state_range_t irep = 0; irep < state.range(index_num_submits); ++irep) {
                for(state_range_t ikernel = 0; ikernel < state.range(index_num_nodes); ++ikernel) {
                    for(state_range_t ibranch = 0; ibranch < state.range(index_num_branches); ++ibranch) {
                        Kokkos::parallel_for(
                            Kokkos::RangePolicy(execs.at(ibranch), 0, 1),
                            Increment{.data = data, .index = ibranch}
                        );
                    }
                }
                for(state_range_t ibranch = 0; ibranch < state.range(index_num_branches); ++ibranch)
                    execs.at(ibranch).fence();
            }
        }
    }

    void run_graph(::benchmark::State& state) const
    {
        Kokkos::utils::timer::Timer<void> timer_A, timer_B;

        double elapsed_create = 0., elapsed_instantiate = 0., elapsed_all_submits = 0., elapsed_per_submit = 0.;

        for (auto sample : state) {
            timer_A.start();
            graph_t graph{Kokkos::Experimental::get_device_handle(execs.at(0))};

            for(state_range_t ibranch = 0; ibranch < state.range(index_num_branches); ++ibranch) {
                auto current = graph.root_node();
                for(state_range_t inode = 0; inode < state.range(index_num_nodes); ++inode) {
                    current = current.then(Increment{.data = data, .index = ibranch});
                }
            }

            timer_A.stop();
            const auto tock_create = timer_A.duration<Kokkos::utils::timer::microseconds>();
            elapsed_create += this->convert(tock_create);

            graph.instantiate();
            timer_A.stop();
            const auto tock_instantiate = timer_A.duration<Kokkos::utils::timer::microseconds>();
            elapsed_instantiate += this->convert(tock_instantiate - tock_create);

            //! Same reason as in @ref benchmarks::graph::StraightLineBenchmark::run_graph.
            graph.submit(execs.at(0));
            execs.at(0).fence();

            timer_B.start();
            for(state_range_t irep = 0; irep < state.range(index_num_submits); ++irep) {
                graph.submit(execs.at(0));
                execs.at(0).fence();
            }
            timer_B.stop();
            timer_A.stop();
            const auto tock_per_submit = timer_B.duration<Kokkos::utils::timer::microseconds>();
            elapsed_per_submit += this->convert(tock_per_submit);
            const auto tock_all_submits = timer_A.duration<Kokkos::utils::timer::microseconds>();
            elapsed_all_submits += this->convert(tock_all_submits - tock_instantiate);
        }

        state.counters["create"] = ::benchmark::Counter(elapsed_create, ::benchmark::Counter::Flags::kAvgIterations);
        state.counters["instantiate"] = ::benchmark::Counter(elapsed_instantiate, ::benchmark::Counter::Flags::kAvgIterations);
        state.counters["per_submit"] = ::benchmark::Counter(elapsed_per_submit / state.range(index_num_submits), ::benchmark::Counter::Flags::kAvgIterations);
        state.counters["all_submits"] = ::benchmark::Counter(elapsed_all_submits, ::benchmark::Counter::Flags::kAvgIterations);
    }

protected:
    std::vector<execution_space> execs;
    view_t data;
};

void parameters(benchmark::Benchmark* benchmark) {
    benchmark->Unit(benchmark::kMicrosecond)
             ->ArgNames({"num_branches", "num_submits", "num_nodes"})
            ->Args({2, 10, 10})
            ->Args({4, 10, 10})
            ->Args({4, 100, 10});
}

BENCHMARK_DEFINE_F(ForkBenchmark, exec)(benchmark::State& state) { this->run_exec(state); }
BENCHMARK_DEFINE_F(ForkBenchmark, graph)(benchmark::State& state) { this->run_graph(state); }
BENCHMARK_REGISTER_F(ForkBenchmark, exec)->Apply(parameters);
BENCHMARK_REGISTER_F(ForkBenchmark, graph)->Apply(parameters);

} // namespace benchmarks::graph
