#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/kokkos_ext/Helpers.hpp"
#include "tests/kokkos_ext/execution_space/Helpers.hpp"
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c bulk by @c Kokkos::Experimental::ExecutionSpaceContext
 * --------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::ExecutionSpaceContext properly customizes
 * @c bulk.
 *
 * The tests can be found in @ref tests/kokkos_ext/execution_space/test_bulk.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext {

using namespace Kokkos::utils::callbacks;

class BulkTest
    : public impl::ExecutionSpaceContextTest<execution_space>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent>;
};

//! @test Check that the chain can be passed by reference to the consumer.
TEST_F(BulkTest, bulk_chain_passed_by_reference) {
    utils::Counter::reset();

    constexpr size_t size = 10;

    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto chain =
        ::stdexec::schedule(esc.get_scheduler())
        | ::stdexec::bulk(::stdexec::par, size, BulkFunctorWithCounter<std::remove_cvref_t<decltype(data)>>(data));

    ASSERT_EQ(utils::Counter::copy_constructions.load(), 0);
    ASSERT_EQ(utils::Counter::move_constructions.load(), 4);

    ::stdexec::sync_wait(chain);

    const size_t expt_copy_constructions = []() {
#if defined(KOKKOS_ENABLE_HPX)
        if constexpr (std::same_as<execution_space, Kokkos::Experimental::HPX>) {
            return 4;
        }
#endif
#if defined(KOKKOS_ENABLE_CUDA)
        if constexpr (std::same_as<execution_space, Kokkos::Cuda>) {
            return 3;
        }
#endif
#if defined(KOKKOS_ENABLE_HIP)
        if constexpr (std::same_as<execution_space, Kokkos::HIP>) {
            return 3;
        }
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
        if constexpr (std::same_as<execution_space, Kokkos::OpenMP>) {
            return 2;
        }
#endif
        throw std::logic_error("Unsupported execution space type.");
    }();

    ASSERT_EQ(utils::Counter::copy_constructions.load(), expt_copy_constructions);
    ASSERT_EQ(utils::Counter::copy_assignments.load(), 0);
    ASSERT_EQ(utils::Counter::move_constructions.load(), 10);
    ASSERT_EQ(utils::Counter::copy_assignments.load(), 0);
}

//! @test Check that @ref Kokkos::Experimental::ExecutionSpaceContext does its duty well when used with @c bulk.
TEST_F(BulkTest, bulk) {
    constexpr size_t size = 10;

    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto chain = ::stdexec::schedule(esc.get_scheduler()) | ADD_BULK(size);

    using chain_t = decltype(chain);

    //! The chain environment advertises the default domain, and completes on the @ref Kokkos::Experimental::details::execution_space::Domain domain.
    static_assert(std::same_as<::stdexec::__domain_of_t<::stdexec::env_of_t<chain_t>>, ::stdexec::default_domain>);
    static_assert(std::same_as<
                  ::stdexec::__detail::__completing_domain_t<::stdexec::set_value_t, chain_t>,
                  Kokkos::Experimental::details::execution_space::Domain
    >);

    //! It has a completion scheduler for the value channel.
    static_assert(std::same_as<
                  decltype(::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(chain))),
                  scheduler_t
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable { ::stdexec::sync_wait(std::move(chain)); }),
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "bulk")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), size / 2 * (size - 1));
}

} // namespace tests::kokkos_ext
