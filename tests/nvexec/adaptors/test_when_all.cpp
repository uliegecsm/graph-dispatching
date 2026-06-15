#include "tests/IgnoreWarnings.hpp"
#include "gtest/gtest.h"

PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED_DEPRECATED_ATTRIBUTES
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsign-compare")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
#include "exec/static_thread_pool.hpp"
#include "nvexec/stream_context.cuh"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/utils/LoadCheckAdd.hpp"

namespace tests::nvexec::adaptors {

class StreamContextTest : public ::testing::Test {
   protected:
    ::nvexec::stream_context stream_ctx{};
};

//! @test Check @c ::stdexec::when_all with all branches on stream scheduler.
TEST_F(StreamContextTest, same_type_children) {
    using value_t = unsigned int;
    using view_t = Kokkos::View<value_t, Kokkos::SharedSpace>;

    const view_t witness(Kokkos::view_alloc("witness"));

    auto schd_stream = stream_ctx.get_scheduler();

    auto when_all = ::stdexec::when_all(
        ::stdexec::schedule(schd_stream) | ::stdexec::then([witness = witness.data()]() {
            if (::nvexec::is_on_gpu())
                Kokkos::atomic_add(witness, 1u);
        }),
        ::stdexec::schedule(schd_stream) | ::stdexec::then([witness = witness.data()]() {
            if (::nvexec::is_on_gpu())
                Kokkos::atomic_add(witness, 10u);
        }));

    ::stdexec::sync_wait(when_all);

    ASSERT_EQ(witness(), 11);
}

//! @test Check @c ::stdexec::when_all with one branch on stream scheduler and one on static thread pool scheduler.
TEST_F(StreamContextTest, different_type_children) {
    ::exec::static_thread_pool thread_ctx{1}; // NOLINT(misc-const-correctness)

    auto schd_thread = thread_ctx.get_scheduler();
    auto schd_stream = stream_ctx.get_scheduler();

    auto domain_thread = ::stdexec::get_completion_domain<::stdexec::set_value_t>(
        ::stdexec::get_env(::stdexec::schedule(schd_thread)), ::stdexec::env<>{});
    static_assert(std::same_as<decltype(domain_thread), ::exec::_pool_::_static_thread_pool::domain>);
    static_assert(std::derived_from<decltype(domain_thread), ::stdexec::default_domain>);

    auto domain_stream = ::stdexec::get_completion_domain<::stdexec::set_value_t>(
        ::stdexec::get_env(::stdexec::schedule(schd_stream)), ::stdexec::env<>{});
    static_assert(std::same_as<decltype(domain_stream), ::nvexec::stream_domain>);
    static_assert(std::derived_from<decltype(domain_stream), ::stdexec::default_domain>);

    auto when_all = ::stdexec::when_all(
        ::stdexec::schedule(schd_stream) | ::stdexec::then([] { }),
        ::stdexec::schedule(schd_thread) | ::stdexec::then([] { }));

    using when_all_t = decltype(when_all);

    auto domain_when_all =
        ::stdexec::get_completion_domain<::stdexec::set_value_t>(::stdexec::get_env(when_all), ::stdexec::env<>{});
    static_assert(std::same_as<
                  decltype(domain_when_all),
                  ::stdexec::indeterminate_domain<::exec::_pool_::_static_thread_pool::domain, ::nvexec::stream_domain>
    >);

    //! The two individual domains are considered default-like domains.
    static_assert(::stdexec::__default_domain_like<
                  ::exec::_pool_::_static_thread_pool::domain,
                  ::stdexec::set_value_t,
                  when_all_t,
                  ::stdexec::env<>
    >);
    static_assert(
        ::stdexec::__default_domain_like<::nvexec::stream_domain, ::stdexec::set_value_t, when_all_t, ::stdexec::env<>>);

    /**
     * The reason is that *e.g.* the stream domain **cannot** transform the @c when_all sender. In this case, the implementation falls back to the default domain.
     * See https://github.com/NVIDIA/stdexec/blob/8cfc3f1983d3521b341864074123281011f998c1/include/stdexec/__detail/__domain.hpp#L93.
     */
    static_assert(!::stdexec::__applicable<
                  ::nvexec::_strm::transform_sender_for<::stdexec::set_value_t>,
                  when_all_t,
                  const ::stdexec::env<>&
    >);

    /**
     * Because the two individual domains are default-like, the customization of the default domain is chosen.
     * See https://github.com/NVIDIA/stdexec/blob/8cfc3f1983d3521b341864074123281011f998c1/include/stdexec/__detail/__domain.hpp#L120-L122.
     */

    /**
     * Calling @c sync_wait on this @c when_all sender triggers a host device error. It appears to be related to the final step where the tuple
     * of values is sent to @c sync_wait receiver.
     * See https://github.com/NVIDIA/stdexec/blob/8cfc3f1983d3521b341864074123281011f998c1/include/stdexec/__detail/__receivers.hpp#L61-L75.
     */
    //::stdexec::sync_wait(when_all);
}

} // namespace tests::nvexec::adaptors
