#ifndef GRAPH_DISPATCHING_UTILS_POOL_HPP
#define GRAPH_DISPATCHING_UTILS_POOL_HPP

#include "Kokkos_Core.hpp"

namespace utils {

/**
 * @brief A pool of @c Kokkos execution space instances.
 *
 * When passing a pool around, consider enqueuing the operations such that it always forms a single-entry
 * single-exit region.
 *
 * That is, ensure that the next scope that uses the pool properly orders if it only uses the first execution space
 * instance of the pool.
 */
template <Kokkos::ExecutionSpace Exec>
struct ExecutionSpacePool {
    using execution_space = Exec;

    std::vector<execution_space> execs;

    explicit ExecutionSpacePool(const size_t size) {
        KOKKOS_EXPECTS(size > 0);
        execs = Kokkos::Experimental::partition_space(Exec{}, std::vector<unsigned short int>(size, 1));
        KOKKOS_ENSURES(execs.size() == size);
    }

    template <std::integral T>
    const Exec& get(const T index) const noexcept {
        KOKKOS_EXPECTS(index < execs.size());
        return execs.at(index);
    }

    auto size() const noexcept { return execs.size(); }
};

//! Convenient fixture with a static size.
template <Kokkos::ExecutionSpace Exec, size_t Size>
struct ExecutionSpacePoolFixture {
    ExecutionSpacePool<Exec> pool{Size};
};

} // namespace utils

#endif // GRAPH_DISPATCHING_UTILS_POOL_HPP
