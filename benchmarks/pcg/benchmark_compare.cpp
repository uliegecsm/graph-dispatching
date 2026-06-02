#include "kokkos-utils/callbacks/EventNameMatcher.hpp"
#include "kokkos-utils/callbacks/Manager.hpp"
#include "kokkos-utils/callbacks/SequenceOfRegionTimerListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "algorithms/cg/Functors.hpp"
#include "algorithms/pcg/Graph.hpp"
#include "algorithms/pcg/Preconditioners.hpp"
#include "algorithms/pcg/SingleQueue.hpp"

#include "tests/cg/Helpers.hpp"

#include "benchmarks/Helpers.hpp"

/**
 * @addtogroup unitbenchmarks
 *
 * Compare PCG implementations
 * ---------------------------
 *
 * These benchmarks compare the performance of the PCG implementations:
 *  - @ref algorithms::pcg::PCGSingleQueue
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

namespace benchmarks::pcg
{

class PCGBenchmark : public benchmarks::BenchmarkBase, public Kokkos::utils::tests::scoped::callbacks::Manager
{
public:
    using pool_t = utils::ExecutionSpacePool<execution_space>;

    static constexpr unsigned short state_nrows   = 0;
    static constexpr unsigned short state_niters  = 1;
    static constexpr unsigned short state_nsweeps = 2;

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
    auto run_once(::benchmark::State& state) const
    {
        auto sequence = std::make_shared<sequence_t>(region_matcher_t{{{
            std::string("PCG") + (std::same_as<T, solver_single_queue_t> ? "SingleQueue" : "Graph") + " - loop"
        }}});
        Kokkos::utils::callbacks::Manager::register_listener(sequence);
        
        const auto [elapsed, res_nrm2, num_iters, sol] = T::run(
            *pool,
            state.range(state_nrows), {.tolerance = tolerance, .max_iters = 20 /*static_cast<size_t>(state.range(state_niters)) * 2*/},
            [num_sweeps = state.range(state_nsweeps)](auto& solver) {
                solver.get_preconditioner().num_sweeps = num_sweeps;
            }
        );

        CHECK_NUMBER_OF_ITERS(static_cast<int64_t>(num_iters), 20 /*state.range(state_niters)*/)

        state.SetIterationTime(std::chrono::duration_cast<Kokkos::utils::timer::seconds>(elapsed).count());

        auto timing = this->convert(sequence->timers.at(0).duration<Kokkos::utils::timer::seconds>());
        state.counters["loop"] = ::benchmark::Counter(timing, ::benchmark::Counter::Flags::kAvgIterations);

        Kokkos::utils::callbacks::Manager::unregister_listener(sequence.get());

        return elapsed;
    }

    void report(::benchmark::State& state, const std::vector<double>& timings) const
    {
        const auto mean = std::accumulate(timings.cbegin(), timings.cend(), 0.) / timings.size();

        state.counters["mean" ] = mean;
        state.counters["nreps"] = timings.size();

        const auto output = std::filesystem::path(CMAKE_CURRENT_BINARY_DIR)
            / state.name()
            / ("nrows_" + std::to_string(state.range(state_nrows)))
            / ("niters_" + std::to_string(state.range(state_niters)))
            / ("nsweeps_" + std::to_string(state.range(state_nsweeps)))
            / "timings.bin";

        this->write(output, std::span{timings});
    }

protected:
    using helper_t = ::tests::cg::NbyNSolverTestHelper<execution_space>;

    using matrix_t  = typename helper_t::initializer_t::matrix_t;
    using rhs_t     = typename helper_t::initializer_t::rhs_t;

    using graph_t = Kokkos::Experimental::Graph<execution_space>;

    using preconditioner_t = ::algorithms::pcg::JacobiPreconditioner<matrix_t>;

    using event_matcher_t  = Kokkos::utils::callbacks::EventNameMatcher;
    using sequence_t       = Kokkos::utils::callbacks::SequenceOfRegionTimerListener<event_matcher_t>;
    using region_matcher_t = typename sequence_t::matcher_t;

    using solver_single_queue_t = ::tests::cg::NbyNSolverTest<::algorithms::pcg::PCGSingleQueue<matrix_t, rhs_t,          preconditioner_t, ::algorithms::cg::Spmv, ::algorithms::cg::Dot, algorithms::cg::Axpby>>;
    using solver_graph_t        = ::tests::cg::NbyNSolverTest<::algorithms::pcg::PCGGraph      <matrix_t, rhs_t, graph_t, preconditioner_t>>;

    std::optional<pool_t> pool = std::nullopt;

    static constexpr typename solver_single_queue_t::solver_t::mag_t tolerance = 1.e-12;
};

#define PCGBENCHMARK_DEFINE_F(_which_)                                 \
    BENCHMARK_DEFINE_F(PCGBenchmark, _which_)(benchmark::State& state) \
    {                                                                  \
        std::vector<double> timings{};                                 \
        for(auto sample : state)                                       \
        {                                                              \
            timings.push_back(this->convert(                           \
                this->run_once<solver_##_which_##_t>(state)));         \
        }                                                              \
        this->report(state, timings);                                  \
    }

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PCGBENCHMARK_REGISTER_F(_which_) BENCHMARK_REGISTER_F(PCGBenchmark, _which_)

void CustomArguments(benchmark::Benchmark* benchmark) {
    benchmark
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)->ArgNames({"nrows", "niters", "nsweeps"})
    //->Args({    10,     4, 4})->Args({    10,     4, 8})->Args({    10,     4, 12})
    //->Args({    20,     9, 4})->Args({    20,     9, 8})->Args({    20,     8, 12})
    //->Args({    50,    23, 4})->Args({    50,    19, 8})->Args({    50,    16, 12})
    //
    //->Args({     100,    43, 4})->Args({     100,    33, 8})->Args({     100,    28, 12})
    //->Args({     150,    62, 4})->Args({     150,    47, 8})->Args({     150,    47, 12})
    //->Args({     200,    81, 4})->Args({     200,    60, 8})->Args({     200,    60, 12})
    //->Args({     300,   118, 4})->Args({     300,    87, 8})->Args({     300,    73, 12})
    //->Args({     500,   190, 4})->Args({     500,   139, 8})->Args({     500,   116, 12})
    //->Args({    1000,   371, 4})->Args({    1000,   268, 8})->Args({    1000,   268, 12})
    //->Args({    2000,   729, 4})->Args({    2000,   522, 8})->Args({    2000,   522, 12})
    //->Args({   10000,  3571, 4})->Args({   10000,  2536, 8})->Args({   10000,  2076, 12})
    //->Args({   20000,  7115, 4})->Args({   20000,  5044, 8})->Args({   20000,  4126, 12})
    //
    //->Args({   50000, 17735, 4})->Args({   50000, 12558, 8})->Args({   50000, 10263, 12})
    //->Args({  100000, 59954, 4})->Args({  100000, 25074, 8})->Args({  100000, 25074, 12})
    //->Args({ 1000000,     0, 4})->Args({ 1000000,     0, 8})->Args({ 1000000,     0, 12})
    ->Args({10000000,     0, 4})->Args({10000000,     0, 8})->Args({10000000,     0, 12})->Args({10000000,     0, 20});
}

PCGBENCHMARK_DEFINE_F(single_queue)
PCGBENCHMARK_DEFINE_F(graph)

PCGBENCHMARK_REGISTER_F(single_queue)->Apply(CustomArguments);
PCGBENCHMARK_REGISTER_F(graph       )->Apply(CustomArguments);

} // namespace benchmarks::pcg
