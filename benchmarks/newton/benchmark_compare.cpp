#include <regex>

#include "plog/Appenders/ColorConsoleAppender.h"
#include "plog/Appenders/RollingFileAppender.h"
#include "plog/Formatters/TxtFormatter.h"
#include "plog/Init.h"
#include "plog/Logger.h"

#include "algorithms/cg/Functors.hpp"
#include "algorithms/newton/Solver.hpp"
#include "algorithms/pcg/Graph.hpp"
#include "algorithms/pcg/Preconditioners.hpp"
#include "algorithms/pcg/SingleQueue.hpp"
#include "apps/heat/NonLinear1DHeatTransfer.hpp"

#include "tests/cg/Helpers.hpp"
#include "tests/newton/Helpers.hpp"

#include "benchmarks/Helpers.hpp"

/**
 * @addtogroup unitbenchmarks
 *
 * Compare Newton implementations
 * ------------------------------
 *
 * These benchmarks compare the performance of @ref algorithms::newton::Solver while using the following
 * linear solvers:
 *  - @ref algorithms::pcg::PCGSingleQueue
 *  - @ref algorithms::pcg::PCGGraph
 *
 * The preconditioner is always @ref algorithms::pcg::JacobiPreconditioner.
 *
 * @todo Investigate for the following scenarios:
 *          - @c HIP
 *          - @c GH200 (or other APUs)
 *
 * The benchmarks can be found in @ref benchmarks/newton/benchmark_compare.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;
using memory_space = typename execution_space::memory_space;

namespace benchmarks::newton {

template <bool UseGraph, bool DetailedCollection>
class NewtonBenchmark : public benchmarks::BenchmarkBase {
   public:
    using problem_t = ::apps::heat::NonLinear1DHeatTransfer<memory_space, execution_space, UseGraph>;

    using pool_t = utils::ExecutionSpacePool<execution_space>;

    using preconditioner_t = ::algorithms::pcg::JacobiPreconditioner<typename problem_t::local_matrix_t>;

    using solver_graph_t = algorithms::pcg::PCGGraph<
        typename problem_t::local_matrix_t,
        typename problem_t::scalar_1d_view_t,
        Kokkos::Experimental::Graph<execution_space>,
        preconditioner_t
    >;

    using solver_single_queue_t = algorithms::pcg::PCGSingleQueue<
        typename problem_t::local_matrix_t,
        typename problem_t::scalar_1d_view_t,
        preconditioner_t,
        ::algorithms::cg::Spmv,
        ::algorithms::cg::Dot,
        ::algorithms::cg::Axpby
    >;

    using linear_solver_t = std::conditional_t<UseGraph, solver_graph_t, solver_single_queue_t>;

    using subtract_t =
        ::tests::newton::Subtract<typename problem_t::scalar_1d_view_t, typename problem_t::scalar_1d_view_t>;

    using newton_t = ::algorithms::newton::Solver<problem_t, linear_solver_t, subtract_t>;

    static constexpr typename problem_t::local_ordinal_t num_elems = 1000;

   public:
    //! We need to create @c Kokkos objects in the @c SetUp, not using the constructor or in-class default member initializers.
    void SetUp(const ::benchmark::State& state) override {
        this->name = std::regex_replace(state.name(), std::regex("[<>,]"), "_");
        this->pool = pool_t{2};

        if constexpr (DetailedCollection) {
            const auto output_dir = std::filesystem::path(CMAKE_CURRENT_BINARY_DIR) / name;
            const auto output_file = output_dir / "output.log";

            std::filesystem::create_directories(output_dir);
            std::filesystem::remove(output_file);

            appender_file = std::make_unique<::plog::RollingFileAppender<::plog::TxtFormatterUtcTime>>(output_file
                                                                                                           .c_str());

            logger = std::make_unique<::plog::Logger<PLOG_DEFAULT_INSTANCE_ID>>(::plog::info);

            logger->addAppender(appender_file.get());
        }
    }

    void TearDown(const ::benchmark::State&) override {
        this->pool = std::nullopt;

        if constexpr (DetailedCollection) {
            logger.reset();
            appender_file.reset();
        }
    }

    FIXME_PARTIAL_OVERRIDE_WARNING_CUDA(SetUp)
    FIXME_PARTIAL_OVERRIDE_WARNING_CUDA(TearDown)

    auto run(::benchmark::State& state) {
        problem_t problem{pool->get(0), typename problem_t::Parameters{.num_elems = num_elems}};
        linear_solver_t linear_solver{pool->get(0), problem.local_matrix, problem.local_rhs};

        linear_solver.get_preconditioner().num_sweeps = 8;

        const newton_t solver{.problem = std::move(problem), .linear_solver = std::move(linear_solver)};

        timer.start();
        [[maybe_unused]] const auto [res_nrm2, num_iters] = solver.solve(
            *pool,
            typename newton_t::Parameters{.tolerance = 1.e-8, .max_iters = std::numeric_limits<size_t>::max()},
            typename linear_solver_t::Parameters{.tolerance = 1.e-8, .max_iters = num_elems});
        timer.stop();

        if constexpr (DetailedCollection) {
            const auto output = std::filesystem::path(CMAKE_CURRENT_BINARY_DIR) / name / "solution.bin";
            const auto sol = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, solver.problem.local_sol);
            using const_value_t = typename std::remove_cvref_t<decltype(sol)>::const_value_type;
            this->write(output, std::span<const_value_t>{sol.data(), sol.size()});
        }

        const auto elapsed = timer.template duration<Kokkos::utils::timer::seconds>();

        state.SetIterationTime(elapsed.count());

        return elapsed;
    }

    void report(::benchmark::State& state, const std::vector<double>& timings) const {
        const auto mean = std::accumulate(timings.cbegin(), timings.cend(), 0.) / timings.size();

        state.counters["mean"] = mean;
        state.counters["nreps"] = timings.size();

        const auto output = std::filesystem::path(CMAKE_CURRENT_BINARY_DIR) / name / "timings.bin";

        this->write(output, std::span{timings});
    }

   protected:
    std::string name;

    std::optional<pool_t> pool = std::nullopt;

    Kokkos::utils::timer::Timer<void> timer;

    std::unique_ptr<::plog::Logger<PLOG_DEFAULT_INSTANCE_ID>> logger = nullptr;

    std::unique_ptr<::plog::RollingFileAppender<::plog::TxtFormatterUtcTime>> appender_file = nullptr;
};

#define NEWTONBENCHMARK(_which_, _use_graph_)                                                                          \
    /* Benchmark that will repeat until the timings have a statistical meaning. */                                     \
    BENCHMARK_TEMPLATE2_DEFINE_F(NewtonBenchmark, repeat##_which_, _use_graph_, false)(benchmark::State & state) {     \
        std::vector<double> timings{};                                                                                 \
        for (auto sample: state) {                                                                                     \
            timings.push_back(this->convert(this->run(state)));                                                        \
        }                                                                                                              \
        this->report(state, timings);                                                                                  \
    }                                                                                                                  \
    BENCHMARK_REGISTER_F(NewtonBenchmark, repeat##_which_)->UseManualTime()->Unit(benchmark::kMillisecond);            \
    /* Benchmark used to collect detailed information about convergence, runs once. */                                 \
    BENCHMARK_TEMPLATE2_DEFINE_F(NewtonBenchmark, collect_##_which_, _use_graph_, true)(benchmark::State & state) {    \
        for (auto sample: state)                                                                                       \
            this->run(state);                                                                                          \
    }                                                                                                                  \
    BENCHMARK_REGISTER_F(NewtonBenchmark, collect_##_which_)->UseManualTime()->Iterations(1);

NEWTONBENCHMARK(single_queue, false)
NEWTONBENCHMARK(graph, true)

} // namespace benchmarks::newton
