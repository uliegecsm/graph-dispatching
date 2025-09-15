#ifndef GRAPH_DISPATCHING_BENCHMARKS_HELPERS_HPP
#define GRAPH_DISPATCHING_BENCHMARKS_HELPERS_HPP

#include <concepts>

#include "benchmark/benchmark.h"

#include "kokkos-utils/timer/Duration.hpp"

namespace benchmarks
{

//! These partial overrides are only needed by 'nvcc' (as of @c Cuda 12.8.1).
#ifdef KOKKOS_ENABLE_CUDA
    //! Use this macro when only one @c __what__ method is overridden.
    #define FIXME_PARTIAL_OVERRIDE_WARNING_CUDA(__what__) \
        void __what__(::benchmark::State& state) override { this->__what__(static_cast<const ::benchmark::State&>(state)); }
#else
    #define FIXME_PARTIAL_OVERRIDE_WARNING_CUDA(...)
#endif

struct BenchmarkBase : public benchmark::Fixture
{
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
};

} // namespace benchmarks

#endif // GRAPH_DISPATCHING_BENCHMARKS_HELPERS_HPP
