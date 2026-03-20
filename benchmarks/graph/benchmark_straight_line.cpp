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
 * Straight-line graph benchmark
 * -----------------------------
 *
 * Compare @c Kokkos::Experimental::Graph with a one-execution-space-instance-based implementation for a
 * straight-line graph.
 *
 * Inspired by:
 *  - https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/
 *
 * The benchmarks can be found in @ref benchmarks/graph/benchmark_straight_line.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace benchmarks::graph
{

template <typename ViewType>
struct Increment {
    ViewType data;

    template <typename... Args>
    KOKKOS_FUNCTION
    void operator()(Args&&...) const noexcept {
        ++data();
    }
};

class StraightLineBenchmark : public benchmarks::BenchmarkBase
{
protected:
    using graph_t = Kokkos::Experimental::Graph<execution_space>;
    using view_t = Kokkos::View<unsigned int, execution_space>;
    using state_range_t = int64_t;

    static constexpr unsigned short int index_num_submits = 0;
    static constexpr unsigned short int index_num_nodes = 1;

public:
    //! We need to create @c Kokkos objects in the @ref SetUp, not using the constructor or in-class default member initializers.
    void SetUp(const ::benchmark::State&) override {
        this->exec = Kokkos::Experimental::partition_space(execution_space{}, 1)[0];
        this->data = view_t(Kokkos::view_alloc(*exec));
    }

    void TearDown(const ::benchmark::State&) override {
        this->data = view_t();
        this->exec = std::nullopt;
    }

    FIXME_PARTIAL_OVERRIDE_WARNING_CUDA(SetUp)
    FIXME_PARTIAL_OVERRIDE_WARNING_CUDA(TearDown)

    void run_exec(::benchmark::State& state) const
    {
        for (auto sample : state) {
            for(state_range_t irep = 0; irep < state.range(index_num_submits); ++irep) {
                for(state_range_t ikernel = 0; ikernel < state.range(index_num_nodes); ++ikernel) {
                    Kokkos::parallel_for(
                        Kokkos::RangePolicy(*exec, 0, 1),
                        Increment{.data = data}
                    );
                }
                exec->fence();
            }
        }
    }

    void run_graph(::benchmark::State& state) const
    {
        Kokkos::utils::timer::Timer<void> timer_A, timer_B;

        double elapsed_create = 0., elapsed_instantiate = 0., elapsed_all_submits = 0., elapsed_per_submit = 0.;

        for (auto sample : state) {
            timer_A.start();
            graph_t graph{Kokkos::Experimental::get_device_handle(*exec)};

            auto current = graph.root_node();

            for(state_range_t inode = 0; inode < state.range(index_num_nodes); ++inode) {
                current = current.then(Increment{.data = data});
            }

            timer_A.stop();
            const auto tock_create = timer_A.duration<Kokkos::utils::timer::microseconds>();
            elapsed_create += this->convert(tock_create);

            graph.instantiate();
            timer_A.stop();
            const auto tock_instantiate = timer_A.duration<Kokkos::utils::timer::microseconds>();
            elapsed_instantiate += this->convert(tock_instantiate - tock_create);

            /// This first submission is not taken into account for the measurement of 'per_submit', because @c CUDA performs a graph upload step
            /// upon the first submission that may not be negligible.
            /// According to https://github.com/ROCm/clr/blob/6dec1eceeb9826c68e0a0b1c114f49ff27efccd2/hipamd/src/hip_graph.cpp#L27-L36,
            /// it seems that @c hipGraphUpload is a no-op.
            /// @c SYCL does not mention such a mechanism.
            graph.submit(*exec);
            exec->fence();

            timer_B.start();
            for(state_range_t irep = 0; irep < state.range(index_num_submits); ++irep) {
                graph.submit(*exec);
                exec->fence();
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
    std::optional<execution_space> exec = std::nullopt;
    view_t data;
};

void parameters(benchmark::Benchmark* benchmark) {
    benchmark->Unit(benchmark::kMicrosecond)
             ->ArgNames({"num_submits", "num_nodes"})
            ->Args({10, 1})
            ->Args({10, 2})
            ->Args({10, 3})
            ->Args({10, 4})
            ->Args({10, 5})
            ->Args({10, 6})
            ->Args({10, 7})
            ->Args({10, 8})
            ->Args({10, 9})
            ->Args({10, 10})
            ->Args({10, 20})
            ->Args({10, 30})
            ->Args({10, 40})
            ->Args({10, 50})
            ->Args({10, 100})
            ->Args({10, 200})
            ->Args({10, 500});
}

BENCHMARK_DEFINE_F(StraightLineBenchmark, exec)(benchmark::State& state) { this->run_exec(state); }
BENCHMARK_DEFINE_F(StraightLineBenchmark, graph)(benchmark::State& state) { this->run_graph(state); }
BENCHMARK_REGISTER_F(StraightLineBenchmark, exec)->Apply(parameters);
BENCHMARK_REGISTER_F(StraightLineBenchmark, graph)->Apply(parameters);

} // namespace benchmarks::graph
