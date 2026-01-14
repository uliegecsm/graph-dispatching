#ifndef GRAPH_DISPATCHING_BENCHMARKS_HELPERS_HPP
#define GRAPH_DISPATCHING_BENCHMARKS_HELPERS_HPP

#include <concepts>
#include <filesystem>
#include <fstream>

#include "benchmark/benchmark.h"

#include "kokkos-utils/timer/Duration.hpp"

namespace benchmarks
{

//! These partial overrides are only needed by 'nvcc' (as of @c Cuda 12.8.1).
#ifdef KOKKOS_ENABLE_CUDA
    //! Use this macro when only one @c __what__ method is overridden. // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
    #define FIXME_PARTIAL_OVERRIDE_WARNING_CUDA(__what__) \
        void __what__(::benchmark::State& state) override { this->__what__(static_cast<const ::benchmark::State&>(state)); }
#else
    #define FIXME_PARTIAL_OVERRIDE_WARNING_CUDA(...)
#endif

template <typename T>
concept BinarySerializable = std::is_trivially_copyable_v<T>;

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

    template <BinarySerializable T>
    static void write(const std::filesystem::path& filename, const std::span<const T> data)
    {
        std::filesystem::create_directories(filename.parent_path());

        std::ofstream out(filename, std::ios::binary);

        if(!out)
            throw std::runtime_error("Failed to open " + filename.string());

        out.write(reinterpret_cast<const char*>(data.data()), data.size_bytes());
    }
};

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CHECK_NUMBER_OF_ITERS(_iters_, _exptd_)                      \
    if(_iters_ != _exptd_)                                           \
    {                                                                \
        std::ostringstream oss;                                      \
        oss << state.name() << ": mismatched number of iterations: " \
            << "expecting " <<  _exptd_ << " but got " << _iters_ ;  \
        state.SkipWithError(oss.str());                              \
    }

} // namespace benchmarks

#endif // GRAPH_DISPATCHING_BENCHMARKS_HELPERS_HPP
