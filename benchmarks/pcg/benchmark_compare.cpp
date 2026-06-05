#include "kokkos-utils/callbacks/EventNameMatcher.hpp"
#include "kokkos-utils/callbacks/Manager.hpp"
#include "kokkos-utils/callbacks/SequenceOfRegionTimerListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "algorithms/cg/Functors.hpp"
#include "algorithms/pcg/Graph.hpp"
#include "algorithms/pcg/Preconditioners.hpp"
#include "algorithms/pcg/Queue.hpp"

#include "tests/cg/Helpers.hpp"

#include "benchmarks/Helpers.hpp"

/**
 * @addtogroup unitbenchmarks
 *
 * Compare PCG implementations
 * ---------------------------
 *
 * These benchmarks compare the performance of the PCG implementations:
 *  - @ref algorithms::pcg::PCGQueue
 *  - @ref algorithms::pcg::PCGGraph
 *
 * The preconditioner is always @ref algorithms::pcg::JacobiPreconditioner.
 *
 * @todo Investigate for the following scenarios:
 *          - @c HIP
 *          - @c GH200 (or other APUs)
 *
 * The benchmarks can be found in @ref benchmarks/pcg/benchmark_compare.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace benchmarks::pcg {

class PCGBenchmark
    : public benchmarks::BenchmarkBase
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using event_matcher_t = Kokkos::utils::callbacks::EventNameMatcher;
    using sequence_t = Kokkos::utils::callbacks::SequenceOfRegionTimerListener<event_matcher_t>;
    using region_matcher_t = typename sequence_t::matcher_t;

    using pool_t = utils::ExecutionSpacePool<execution_space>;

    //! Cap the number of iterations.
    static constexpr size_t capped_max_iters = 10;

    static constexpr unsigned short state_nrows = 0;
    static constexpr unsigned short state_nsweeps = 1;

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

    template <typename T>
    auto run(::benchmark::State& state) {
        std::vector<double> timings;

        if constexpr (std::same_as<T, solver_queue_t>) {
            sequence = std::make_shared<sequence_t>(region_matcher_t{{{"PCGQueue - loop"}}});
            timings.resize(1);
        } else {
            sequence = std::make_shared<sequence_t>(
                region_matcher_t{{{"PCGGraph - graph definition"}}},
                region_matcher_t{{{"PCGGraph - graph instantiation"}}},
                region_matcher_t{{{"PCGGraph - submit 0"}}},
                region_matcher_t{{{"PCGGraph - submit r"}}});
            timings.resize(4);
        }

        Kokkos::utils::callbacks::Manager::register_listener(sequence);

        size_t num_iters_sum = 0;

        for (auto sample: state) {
            sequence->reset();

            const auto [elapsed, res_nrm2, num_iters, sol] = T::run(
                *pool,
                state.range(state_nrows),
                {.tolerance = tolerance, .max_iters = capped_max_iters},
                [num_sweeps = state.range(state_nsweeps)](auto& solver) {
                    solver.get_preconditioner().num_sweeps = num_sweeps;
                });

            num_iters_sum += num_iters;

            CHECK_NUMBER_OF_ITERS(num_iters, >, capped_max_iters)

            for (size_t itiming = 0; itiming < timings.size(); ++itiming) {
                timings.at(itiming) += this->convert(sequence->timers.at(itiming)
                                                         .duration<Kokkos::utils::timer::microseconds>());
            }

            state.SetIterationTime(std::chrono::duration_cast<Kokkos::utils::timer::seconds>(elapsed).count());
        }

        Kokkos::utils::callbacks::Manager::unregister_listener(sequence.get());

        if constexpr (std::same_as<T, solver_queue_t>) {
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

    using graph_t = Kokkos::Experimental::Graph<execution_space>;

    using preconditioner_t = ::algorithms::pcg::JacobiPreconditioner<matrix_t>;

    using solver_queue_t = ::tests::cg::NbyNSolverTest<::algorithms::pcg::PCGQueue<
        matrix_t,
        rhs_t,
        preconditioner_t,
        ::algorithms::cg::Spmv,
        ::algorithms::cg::Dot,
        algorithms::cg::Axpby
    >>;
    using solver_graph_t =
        ::tests::cg::NbyNSolverTest<::algorithms::pcg::PCGGraph<matrix_t, rhs_t, graph_t, preconditioner_t>>;

    std::optional<pool_t> pool = std::nullopt;

    std::shared_ptr<sequence_t> sequence = nullptr;

    static constexpr typename solver_queue_t::solver_t::mag_t tolerance = 1.e-12;
};

#define PCGBENCHMARK_DEFINE_F(_which_)                                                                                 \
    BENCHMARK_DEFINE_F(PCGBenchmark, _which_)(benchmark::State & state) {                                              \
        this->run<solver_##_which_##_t>(state);                                                                        \
    }
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PCGBENCHMARK_REGISTER_F(_which_) BENCHMARK_REGISTER_F(PCGBenchmark, _which_)

void CustomArguments(benchmark::Benchmark* benchmark) {
    benchmark->UseManualTime()
        ->Unit(benchmark::kMillisecond)
        ->ArgNames({"nrows", "nsweeps"})

        ->Args({10, 4})
        ->Args({10, 8})
        ->Args({10, 12})
        ->Args({20, 4})
        ->Args({20, 8})
        ->Args({20, 12})
        ->Args({50, 4})
        ->Args({50, 8})
        ->Args({50, 12})
#if false
        ->ArgsProduct({
            benchmark::CreateRange(128, 128<<15, 2),
            {4, 8, 12}
        })
#endif
        ;
}

PCGBENCHMARK_DEFINE_F(queue)
PCGBENCHMARK_DEFINE_F(graph)

PCGBENCHMARK_REGISTER_F(queue)->Apply(CustomArguments);
PCGBENCHMARK_REGISTER_F(graph)->Apply(CustomArguments);

} // namespace benchmarks::pcg
