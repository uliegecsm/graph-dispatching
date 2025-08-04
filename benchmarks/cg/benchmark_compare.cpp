#include "benchmark/benchmark.h"

#include "kokkos-utils/callbacks/EventNameMatcher.hpp"
#include "kokkos-utils/callbacks/Manager.hpp"
#include "kokkos-utils/callbacks/SequenceOfRegionTimerListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "algorithms/cg/Graph.hpp"
#include "algorithms/cg/SingleQueue.hpp"

#include "tests/cg/Helpers.hpp"

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
class CGBenchmark : public benchmark::Fixture,
                    public Kokkos::utils::tests::scoped::callbacks::Manager
{
protected:
    using event_matcher_t  = Kokkos::utils::callbacks::EventNameMatcher;
    using sequence_t       = Kokkos::utils::callbacks::SequenceOfRegionTimerListener<event_matcher_t>;
    using region_matcher_t = typename sequence_t::matcher_t;

public:
    //! We need to create @c Kokkos objects in the @c SetUp, not using the constructor or in-class default member initializers.
    void SetUp(const ::benchmark::State&) override {
        this->exec = Kokkos::Experimental::partition_space(execution_space{}, 1)[0];
    }

    void TearDown(const ::benchmark::State&) override {
        this->exec = std::nullopt;
    }

//! These partial overrides are only needed by 'nvcc' (as of @c Cuda 12.8.1).
#ifdef KOKKOS_ENABLE_CUDA
    //! Use this macro when only one @c __what__ method is overridden.
    #define FIXME_PARTIAL_OVERRIDE_WARNING_CUDA(__what__) \
        void __what__(::benchmark::State& state) override { this->__what__(static_cast<const ::benchmark::State&>(state)); }
#else
    #define FIXME_PARTIAL_OVERRIDE_WARNING_CUDA(...)
#endif

    FIXME_PARTIAL_OVERRIDE_WARNING_CUDA(SetUp)
    FIXME_PARTIAL_OVERRIDE_WARNING_CUDA(TearDown)

    //! Convert @p from to the time unit used by the benchmark.
    template <typename Duration>
    auto convert(const Duration& duration) const
    {
        switch(this->GetTimeUnit())
        {
            case ::benchmark::kMicrosecond: return std::chrono::duration_cast<Kokkos::utils::timer::microseconds>(duration).count();
            case ::benchmark::kMillisecond: return std::chrono::duration_cast<Kokkos::utils::timer::milliseconds>(duration).count();
            case ::benchmark::kSecond:      return std::chrono::duration_cast<Kokkos::utils::timer::seconds     >(duration).count();
            default:
                Kokkos::abort("unsupported time unit");
        }
    }

    #define CHECK_SEQUENCE_AND_NUM_ITERS(_seq_, _iters_, _exptd_)        \
        if(!_seq_->all_matched())                                        \
            Kokkos::abort("At least one sub-region did not match.");     \
        if(_iters_ != _exptd_)                                           \
        {                                                                \
            std::ostringstream oss;                                      \
            oss << state.name() << ": mismatched number of iterations: " \
                << _iters_ << " != " << _exptd_;                         \
            state.SkipWithError(oss.str());                              \
        }

    template <typename T>
    void run_upto_convergence(::benchmark::State& state, std::array<double, 1>& timings) const requires (CollectionMethodValue == CollectionMethod::CONVERGENCE)
    {
        sequence->reset();

        const auto [elapsed, res_nrm2, num_iters, sol] = T::run(*exec, state.range(0), tolerance, state.range(1) * 2);

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

        const auto [elapsed, res_nrm2, num_iters, sol] = T::run(*exec, state.range(0), tolerance, 1);

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

    std::optional<execution_space> exec = std::nullopt;

    std::shared_ptr<sequence_t> sequence = nullptr;

    static constexpr typename single_queue_t::solver_t::mag_t tolerance = 1.e-12;
};

#define CGBENCHMARK_DEFINE_UPTO_CONVERGENCE_F(_which_) BENCHMARK_TEMPLATE_DEFINE_F(CGBenchmark, _which_##convergence, CollectionMethod::CONVERGENCE)(benchmark::State& state) { this->run<_which_##_t>(state); }
#define CGBENCHMARK_DEFINE_UPTO_SINGLE_ITER_F(_which_) BENCHMARK_TEMPLATE_DEFINE_F(CGBenchmark, _which_##single_iter, CollectionMethod::SINGLE_ITER)(benchmark::State& state) { this->run<_which_##_t>(state); }

#define CGBENCHMARK_REGISTER_UPTO_CONVERGENCE_F(_which_) BENCHMARK_REGISTER_F(CGBenchmark, _which_##convergence)
#define CGBENCHMARK_REGISTER_UPTO_SINGLE_ITER_F(_which_) BENCHMARK_REGISTER_F(CGBenchmark, _which_##single_iter)

void CustomArguments(benchmark::internal::Benchmark* benchmark) {
    benchmark
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)->ArgNames({"nrows", "niters"})
    ->Args({   10,     9})
    ->Args({   20,    18})
    ->Args({   90,    89});
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
