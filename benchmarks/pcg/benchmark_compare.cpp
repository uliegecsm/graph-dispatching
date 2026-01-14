#include <format>

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

class PCGBenchmark : public benchmarks::BenchmarkBase
{
public:
    static constexpr unsigned short state_nrows   = 0;
    static constexpr unsigned short state_niters  = 1;
    static constexpr unsigned short state_nsweeps = 2;

public:
    //! We need to create @c Kokkos objects in the @c SetUp, not using the constructor or in-class default member initializers.
    void SetUp(const ::benchmark::State&) override {
        this->exec = Kokkos::Experimental::partition_space(execution_space{}, 1)[0];
    }

    void TearDown(const ::benchmark::State&) override {
        this->exec = std::nullopt;
    }

    FIXME_PARTIAL_OVERRIDE_WARNING_CUDA(SetUp)
    FIXME_PARTIAL_OVERRIDE_WARNING_CUDA(TearDown)

    template <typename T>
    auto run_once(::benchmark::State& state) const
    {
        const auto [elapsed, res_nrm2, num_iters, sol] = T::run(
            *exec,
            state.range(state_nrows), {.tolerance = tolerance, .max_iters = static_cast<size_t>(state.range(state_niters)) * 2},
            [num_sweeps = state.range(state_nsweeps)](auto& solver) {
                solver.get_preconditioner().num_sweeps = num_sweeps;
            }
        );

        CHECK_NUMBER_OF_ITERS(static_cast<int64_t>(num_iters), state.range(state_niters))

        state.SetIterationTime(std::chrono::duration_cast<Kokkos::utils::timer::seconds>(elapsed).count());

        return elapsed;
    }

    void report(::benchmark::State& state, const std::vector<double>& timings) const
    {
        const auto mean = std::accumulate(timings.cbegin(), timings.cend(), 0.) / timings.size();

        state.counters["mean" ] = mean;
        state.counters["nreps"] = timings.size();

        const auto output = std::filesystem::path(CMAKE_CURRENT_BINARY_DIR)
            / state.name()
            / std::format("nrows_{}",   state.range(state_nrows))
            / std::format("niters_{}",  state.range(state_niters))
            / std::format("nsweeps_{}", state.range(state_nsweeps))
            / "timings.bin";

        this->write(output, std::span{timings});
    }

protected:
    using helper_t = ::tests::cg::NbyNSolverTestHelper<execution_space>;

    using matrix_t  = typename helper_t::initializer_t::matrix_t;
    using rhs_t     = typename helper_t::initializer_t::rhs_t;

    using graph_t = Kokkos::Experimental::Graph<execution_space>;

    using preconditioner_t = ::algorithms::pcg::JacobiPreconditioner<matrix_t>;

    using solver_single_queue_t = ::tests::cg::NbyNSolverTest<::algorithms::pcg::PCGSingleQueue<matrix_t, rhs_t,          preconditioner_t, ::algorithms::cg::Spmv, ::algorithms::cg::Dot, algorithms::cg::Axpby>>;
    using solver_graph_t        = ::tests::cg::NbyNSolverTest<::algorithms::pcg::PCGGraph      <matrix_t, rhs_t, graph_t, preconditioner_t>>;

    std::optional<execution_space> exec = std::nullopt;

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

void CustomArguments(benchmark::internal::Benchmark* benchmark) {
    benchmark
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)->ArgNames({"nrows", "niters", "nsweeps"})

    ->Args({10,  4, 4})
    ->Args({10,  4, 8})

    ->Args({20,  9, 4})
    ->Args({20,  9, 8})

    ->Args({40,  19, 4})
    ->Args({40,  16, 8})
    ;
}

PCGBENCHMARK_DEFINE_F(single_queue)
PCGBENCHMARK_DEFINE_F(graph)

PCGBENCHMARK_REGISTER_F(single_queue)->Apply(CustomArguments);
PCGBENCHMARK_REGISTER_F(graph       )->Apply(CustomArguments);

} // namespace benchmarks::pcg
