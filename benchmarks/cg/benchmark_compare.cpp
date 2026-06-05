#include "benchmark/benchmark.h"

#include "kokkos-utils/callbacks/EventNameMatcher.hpp"
#include "kokkos-utils/callbacks/Manager.hpp"
#include "kokkos-utils/callbacks/SequenceOfRegionTimerListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "algorithms/cg/Graph.hpp"
#include "algorithms/cg/Queue.hpp"

#include "tests/cg/Helpers.hpp"

#include "benchmarks/Helpers.hpp"

#if defined(KOKKOS_ENABLE_SYCL) && !defined(KOKKOS_IMPL_SYCL_GRAPH_SUPPORT)
#    error "KOKKOS_IMPL_SYCL_GRAPH_SUPPORT is required."
#endif

/**
 * @addtogroup unitbenchmarks
 *
 * Compare CG implementations
 * --------------------------
 *
 * These benchmarks compare the performance of the CG implementations:
 *  - @ref algorithms::cg::CGQueue
 *  - @ref algorithms::cg::CGGraph with host nodes
 *  - @ref algorithms::cg::CGGraph
 *
 * As of @c Cuda 12.8.1, it seems that using the device @c then node is always better.
 *
 * @todo Investigate for the following scenarios:
 *          - @c HIP
 *          - @c GH200 (or other APUs)
 *
 * The benchmarks can be found in @ref benchmarks/cg/benchmark_compare.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace benchmarks::cg {

class CGBenchmark
    : public benchmarks::BenchmarkBase
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   protected:
    using event_matcher_t = Kokkos::utils::callbacks::EventNameMatcher;
    using sequence_t = Kokkos::utils::callbacks::SequenceOfRegionTimerListener<event_matcher_t>;
    using region_matcher_t = typename sequence_t::matcher_t;

    using pool_t = utils::ExecutionSpacePool<execution_space>;

    //! Cap the number of iterations.
    static constexpr size_t capped_max_iters = 10;

    static constexpr unsigned short int index_num_rows = 0;

   public:
    //! We need to create @c Kokkos objects in the @c SetUp, not using the constructor or in-class default member initializers.
    void SetUp(const ::benchmark::State&) override {
        this->pool = pool_t{2};
    }

    void TearDown(const ::benchmark::State&) override {
        this->pool = std::nullopt;
    }

    FIXME_PARTIAL_OVERRIDE_WARNING_CUDA(SetUp)
    FIXME_PARTIAL_OVERRIDE_WARNING_CUDA(TearDown)

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CHECK_SEQUENCE(_seq_)                                                                                          \
    if (!_seq_->all_matched())                                                                                         \
        Kokkos::abort("At least one sub-region did not match.");

    template <typename T>
    void run(::benchmark::State& state) {
        std::vector<double> timings;

        if constexpr (std::same_as<T, queue_t>) {
            sequence = std::make_shared<sequence_t>(region_matcher_t{{{"CGQueue - loop"}}});
            timings.resize(1);
        } else {
            sequence = std::make_shared<sequence_t>(
                region_matcher_t{{{"CGGraph - graph definition"}}},
                region_matcher_t{{{"CGGraph - graph instantiation"}}},
                region_matcher_t{{{"CGGraph - submit 0"}}},
                region_matcher_t{{{"CGGraph - submit r"}}});
            timings.resize(4);
        }

        Kokkos::utils::callbacks::Manager::register_listener(sequence);

        size_t num_iters_sum = 0;

        for (auto sample: state) {
            sequence->reset();

            const auto [elapsed, res_nrm2, num_iters, sol] = T::run(
                *pool,
                state.range(index_num_rows),
                typename T::solver_t::Parameters{.tolerance = tolerance, .max_iters = capped_max_iters});

            num_iters_sum += num_iters;

            CHECK_SEQUENCE(sequence)
            CHECK_NUMBER_OF_ITERS(num_iters, >, capped_max_iters)

            for (size_t itiming = 0; itiming < timings.size(); ++itiming) {
                timings.at(itiming) += this->convert(sequence->timers.at(itiming)
                                                         .duration<Kokkos::utils::timer::microseconds>());
            }

            state.SetIterationTime(std::chrono::duration_cast<Kokkos::utils::timer::seconds>(elapsed).count());
        }

        Kokkos::utils::callbacks::Manager::unregister_listener(sequence.get());

        if constexpr (std::same_as<T, queue_t>) {
            state.counters["loop"] = ::benchmark::Counter(timings.at(0), ::benchmark::Counter::Flags::kAvgIterations);
        } else {
            state.counters["graph definition"] =
                ::benchmark::Counter(timings.at(0), ::benchmark::Counter::Flags::kAvgIterations);
            state.counters["graph instantiation"] =
                ::benchmark::Counter(timings.at(1), ::benchmark::Counter::Flags::kAvgIterations);
            state.counters["graph submit 0"] =
                ::benchmark::Counter(timings.at(2), ::benchmark::Counter::Flags::kAvgIterations);
            state.counters["graph submit r"] =
                ::benchmark::Counter(timings.at(3), ::benchmark::Counter::Flags::kAvgIterations);
        }

        state.counters["num_iters"] = ::benchmark::Counter(num_iters_sum, ::benchmark::Counter::Flags::kAvgIterations);
    }

   protected:
    using helper_t = ::tests::cg::NbyNSolverTestHelper<execution_space>;

    using matrix_t = typename helper_t::initializer_t::matrix_t;
    using rhs_t = typename helper_t::initializer_t::rhs_t;

    using queue_t = ::tests::cg::NbyNSolverTest<
        ::algorithms::cg::CGQueue<matrix_t, rhs_t, ::algorithms::cg::Spmv, ::algorithms::cg::Dot, algorithms::cg::Axpby>
    >;
    using graph_t = ::tests::cg::NbyNSolverTest<::algorithms::cg::CGGraph<matrix_t, rhs_t, false>>;
    using graph_with_host_t = ::tests::cg::NbyNSolverTest<::algorithms::cg::CGGraph<matrix_t, rhs_t, true>>;

    std::optional<pool_t> pool = std::nullopt;

    std::shared_ptr<sequence_t> sequence = nullptr;

    static constexpr typename queue_t::solver_t::mag_t tolerance = 1.e-12;
};

#define CGBENCHMARK_DEFINE_F(_which_)                                                                                  \
    BENCHMARK_DEFINE_F(CGBenchmark, _which_)(benchmark::State & state) {                                               \
        this->run<_which_##_t>(state);                                                                                 \
    }
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CGBENCHMARK_REGISTER_F(_which_) BENCHMARK_REGISTER_F(CGBenchmark, _which_)

void parameters(benchmark::Benchmark* benchmark) {
    benchmark->UseManualTime()
        ->Unit(benchmark::kMillisecond)
        ->ArgName("num_rows")
        ->Arg(10)
        ->Arg(20)
        ->Arg(30)
        ->Arg(50)
        ->Arg(60)
        ->Arg(70)
#if false
        ->RangeMultiplier(2)->Range(128, 128<<15)
#endif
        ;
}

CGBENCHMARK_DEFINE_F(queue)
CGBENCHMARK_DEFINE_F(graph)
CGBENCHMARK_DEFINE_F(graph_with_host)

CGBENCHMARK_REGISTER_F(queue)->Apply(parameters);
CGBENCHMARK_REGISTER_F(graph)->Apply(parameters);
CGBENCHMARK_REGISTER_F(graph_with_host)->Apply(parameters);

} // namespace benchmarks::cg
