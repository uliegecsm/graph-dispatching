#include "benchmark/benchmark.h"

#include "kokkos-utils/callbacks/EventNameMatcher.hpp"
#include "kokkos-utils/callbacks/Manager.hpp"
#include "kokkos-utils/callbacks/SequenceOfRegionTimerListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "algorithms/cg/Graph.hpp"
#include "algorithms/cg/SingleQueue.hpp"

#include "tests/cg/Helpers.hpp"

#include "benchmarks/Helpers.hpp"

#if defined(KOKKOS_ENABLE_SYCL) && !defined(KOKKOS_IMPL_SYCL_GRAPH_SUPPORT)
    #error "KOKKOS_IMPL_SYCL_GRAPH_SUPPORT is required."
#endif

/**
 * @addtogroup unitbenchmarks
 *
 * Compare CG implementations
 * --------------------------
 *
 * These benchmarks compare the performance of the CG implementations:
 *  - @ref algorithms::cg::CGSingleQueue
 *  - @ref algorithms::cg::CGGraph with host nodes
 *  - @ref algorithms::cg::CGGraph
 *
 * As of @c Cuda 12.8.1, it seems that using the device then node is always better.
 *
 * @todo Investigate for the following scenarios:
 *          - @c HIP
 *          - @c GH200 (or other APUs)
 *
 * The benchmarks can be found in @ref benchmarks/cg/benchmark_compare.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace benchmarks::cg
{

enum class CollectionMethod : std::uint8_t 
{
    CONVERGENCE, //!< Run the solver until convergence, disable sub-region timers.
    SINGLE_ITER  //!< Run one iteration of the solver, enable sub-region timers.
};

template <CollectionMethod CollectionMethodValue>
class CGBenchmark : public benchmarks::BenchmarkBase,
                    public Kokkos::utils::tests::scoped::callbacks::Manager
{
protected:
    using event_matcher_t  = Kokkos::utils::callbacks::EventNameMatcher;
    using sequence_t       = Kokkos::utils::callbacks::SequenceOfRegionTimerListener<event_matcher_t>;
    using region_matcher_t = typename sequence_t::matcher_t;

    using pool_t = utils::ExecutionSpacePool<execution_space>;

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
    #define CHECK_SEQUENCE_AND_NUM_ITERS(_seq_, _iters_, _exptd_)        \
        if(!_seq_->all_matched())                                        \
            Kokkos::abort("At least one sub-region did not match.");     \
        CHECK_NUMBER_OF_ITERS(_iters_, _exptd_)

    template <typename T>
    void run_upto_convergence(::benchmark::State& state, std::array<double, 1>& timings) const requires (CollectionMethodValue == CollectionMethod::CONVERGENCE)
    {
        sequence->reset();

        const auto [elapsed, res_nrm2, num_iters, sol] = T::run(*pool, state.range(0), typename T::solver_t::Parameters{.tolerance = tolerance, .max_iters = static_cast<size_t>(state.range(1) * 2)});

        CHECK_SEQUENCE_AND_NUM_ITERS(sequence, num_iters, static_cast<size_t>(state.range(1)))

        for(size_t itiming = 0; itiming < timings.size(); ++itiming) {
            timings.at(itiming) += this->convert(sequence->timers.at(itiming).duration<Kokkos::utils::timer::microseconds>());
        }

        state.SetIterationTime(std::chrono::duration_cast<Kokkos::utils::timer::seconds>(elapsed).count());
    }

    template <typename T>
    void run_upto_single_iter(::benchmark::State& state, std::array<double, 3>& timings) const requires (CollectionMethodValue == CollectionMethod::SINGLE_ITER)
    {
        sequence->reset();

        const auto [elapsed, res_nrm2, num_iters, sol] = T::run(*pool, state.range(0), typename T::solver_t::Parameters{.tolerance = tolerance, .max_iters = 1});

        CHECK_SEQUENCE_AND_NUM_ITERS(sequence, num_iters, 1)

        Kokkos::utils::timer::seconds accumulated{0.};

        for(size_t itiming = 0; itiming < timings.size(); ++itiming) {
            accumulated         +=               sequence->timers.at(itiming).duration<Kokkos::utils::timer::     seconds>();
            timings.at(itiming) += this->convert(sequence->timers.at(itiming).duration<Kokkos::utils::timer::microseconds>());
        }

        state.SetIterationTime(accumulated.count());
    }

    template <typename T>
    void run(::benchmark::State& state) requires (CollectionMethodValue == CollectionMethod::CONVERGENCE)
    {
        sequence = std::make_shared<sequence_t>(region_matcher_t{{{
            std::string("CG") + (std::same_as<T, single_queue_t> ? "SingleQueue" : "Graph") + " - loop"
        }}});
        Kokkos::utils::callbacks::Manager::register_listener(sequence);

        std::array<double, 1> timings{0.};

        for (auto sample : state)
            this->run_upto_convergence<T>(state, timings);

        state.counters["loop"] = ::benchmark::Counter(timings.at(0), ::benchmark::Counter::Flags::kAvgIterations);

        Kokkos::utils::callbacks::Manager::unregister_listener(sequence.get());
    }

    template <typename T>
    void run(::benchmark::State& state) requires (CollectionMethodValue == CollectionMethod::SINGLE_ITER)
    {
        sequence = std::make_shared<sequence_t>(
            region_matcher_t{{{"CGGraph - setup"}}},
            region_matcher_t{{{"CGGraph - create graph"}}},
            region_matcher_t{{{"CGGraph - instantiate graph"}}}
        );
        Kokkos::utils::callbacks::Manager::register_listener(sequence);

        std::array<double, 3> timings{0., 0., 0.};

        for (auto sample : state) {
            this->run_upto_single_iter<T>(state, timings);
        }
        state.counters["setup"]             = ::benchmark::Counter(timings.at(0), ::benchmark::Counter::Flags::kAvgIterations);
        state.counters["create-graph"]      = ::benchmark::Counter(timings.at(1), ::benchmark::Counter::Flags::kAvgIterations);
        state.counters["instantiate-graph"] = ::benchmark::Counter(timings.at(2), ::benchmark::Counter::Flags::kAvgIterations);

        Kokkos::utils::callbacks::Manager::unregister_listener(sequence.get());
    }

protected:
    using helper_t = ::tests::cg::NbyNSolverTestHelper<execution_space>;

    using matrix_t  = typename helper_t::initializer_t::matrix_t;
    using rhs_t     = typename helper_t::initializer_t::rhs_t;

    using single_queue_t    = ::tests::cg::NbyNSolverTest<::algorithms::cg::CGSingleQueue<matrix_t, rhs_t, ::algorithms::cg::Spmv, ::algorithms::cg::Dot, algorithms::cg::Axpby>>;
    using graph_t           = ::tests::cg::NbyNSolverTest<::algorithms::cg::CGGraph      <matrix_t, rhs_t, false>>;
    using graph_with_host_t = ::tests::cg::NbyNSolverTest<::algorithms::cg::CGGraph      <matrix_t, rhs_t, true>>;

    std::optional<pool_t> pool = std::nullopt;

    std::shared_ptr<sequence_t> sequence = nullptr;

    static constexpr typename single_queue_t::solver_t::mag_t tolerance = 1.e-12;
};

#define CGBENCHMARK_DEFINE_UPTO_CONVERGENCE_F(_which_) BENCHMARK_TEMPLATE_DEFINE_F(CGBenchmark, _which_##convergence, CollectionMethod::CONVERGENCE)(benchmark::State& state) { this->run<_which_##_t>(state); }
#define CGBENCHMARK_DEFINE_UPTO_SINGLE_ITER_F(_which_) BENCHMARK_TEMPLATE_DEFINE_F(CGBenchmark, _which_##single_iter, CollectionMethod::SINGLE_ITER)(benchmark::State& state) { this->run<_which_##_t>(state); }

#define CGBENCHMARK_REGISTER_UPTO_CONVERGENCE_F(_which_) BENCHMARK_REGISTER_F(CGBenchmark, _which_##convergence)
#define CGBENCHMARK_REGISTER_UPTO_SINGLE_ITER_F(_which_) BENCHMARK_REGISTER_F(CGBenchmark, _which_##single_iter)

void CustomArguments(benchmark::Benchmark* benchmark) {
    benchmark
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)->ArgNames({"nrows", "niters"})
    ->Args({    10,     9})
    ->Args({    20,    18})
    ->Args({    50,    48})
#if false
    ->Args({   100,    99})
    ->Args({   150,   149})
    ->Args({   200,   198})
    ->Args({   300,   299})
    ->Args({   500,   498})
    ->Args({  1000,   999})
    ->Args({  2000,  1998})
    ->Args({ 10000,  9999})
    ->Args({ 20000, 19998})
    ->Args({ 50000, 49998})
    ->Args({100000, 99999})
#endif
;
}

/// @name Benchmarks that will go up to convergence.
///@{
CGBENCHMARK_DEFINE_UPTO_CONVERGENCE_F(single_queue)
CGBENCHMARK_DEFINE_UPTO_CONVERGENCE_F(graph)
CGBENCHMARK_DEFINE_UPTO_CONVERGENCE_F(graph_with_host)

CGBENCHMARK_REGISTER_UPTO_CONVERGENCE_F(single_queue   )->Apply(CustomArguments);
CGBENCHMARK_REGISTER_UPTO_CONVERGENCE_F(graph          )->Apply(CustomArguments);
CGBENCHMARK_REGISTER_UPTO_CONVERGENCE_F(graph_with_host)->Apply(CustomArguments);
///@}

/// @name Benchmarks that will do a single iteration.
///@{
CGBENCHMARK_DEFINE_UPTO_SINGLE_ITER_F(graph)
CGBENCHMARK_DEFINE_UPTO_SINGLE_ITER_F(graph_with_host)

CGBENCHMARK_REGISTER_UPTO_SINGLE_ITER_F(graph          )->Apply(CustomArguments);
CGBENCHMARK_REGISTER_UPTO_SINGLE_ITER_F(graph_with_host)->Apply(CustomArguments);
///@}
} // namespace benchmarks::cg
