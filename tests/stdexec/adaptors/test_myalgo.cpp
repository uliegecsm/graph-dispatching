#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos_ext/impl/completion_signatures.hpp"

#include "tests/stdexec/Utils.hpp"
#include "tests/stdexec/adaptors/myalgo.hpp"
#include "tests/stdexec/inline_scheduler_with_domain.hpp"

/**
 * Add a new algorithm to the customizable set
 * -------------------------------------------
 *
 * This group of tests show how one can add a new, customizable algorithm.
 *
 * The new algorithm implementation can be found in @ref tests/stdexec/adaptors/myalgo.hpp.
 *
 * The tests can be found in @ref tests/stdexec/adaptors/test_myalgo.cpp.
 */

namespace some_other_namespace {

//! Custom implementation of @ref myalgo_namespace::myalgo_t for @ref iswd::domain.
template <::stdexec::sender Sndr, typename Obj>
struct custom_printf_sender {
    using sender_concept = ::stdexec::sender_t;

    Sndr sndr;
    Obj obj;

    template <::stdexec::receiver DownstreamRcvr>
    struct receiver {
        using receiver_concept = ::stdexec::receiver_t;

        DownstreamRcvr downstream_rcvr;
        Obj obj;

        template <typename... Args>
        void set_value(Args&&... args) && noexcept {
            std::cout << "MY BIG FAT CUSTOM PREFIX: " << obj << '\n';
            ::stdexec::set_value(std::move(downstream_rcvr), std::forward<Args>(args)...);
        }
    };

    //! Keep the same completion signatures as the upstream sender, because it will forward anything.
    GRAPH_DISPATCHING_KOKKOS_EXT_COMPL_SIGS_KEEP(custom_printf_sender)

    template <::stdexec::receiver Rcvr>
    ::stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
        return ::stdexec::connect(
            std::move(sndr),
            receiver<std::remove_cvref_t<Rcvr>>{.downstream_rcvr = std::forward<Rcvr>(rcvr), .obj = std::move(obj)});
    }
};

} // namespace some_other_namespace

namespace iswd {

template <typename Env>
struct transform_sender_for<myalgo_namespace::myalgo_t, Env> {
    template <typename Obj, ::stdexec::sender Sndr>
    auto operator()(myalgo_namespace::myalgo_t, Obj&& obj, Sndr&& sndr) && noexcept {
        return some_other_namespace::custom_printf_sender<std::remove_cvref_t<Sndr>, std::remove_cvref_t<Obj>>{
            .sndr = std::forward<Sndr>(sndr), .obj = std::forward<Obj>(obj)};
    }

    const Env& env_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

} // namespace iswd

namespace tests::stdexec::adaptors {

struct Printable {
    double value;

    friend std::ostream& operator<<(std::ostream& out, const Printable& obj) {
        return out << "Printable(value=" << obj.value << ")";
    }
};

//! @test Check that @ref myalgo_namespace::myalgo works as expected (no customization).
TEST(myalgo_namespace, myalgo) {
    ::testing::internal::CaptureStdout();
    {
        auto work = ::stdexec::just(Printable{.value = 42})
                  | myalgo_namespace::myalgo("Hello from default implementation!")
                  | ::stdexec::then([](const Printable& value) {
                        std::cout << "Waving from 'then' with value " << value << ".\n";
                    });
        ::stdexec::sync_wait(std::move(work)); // NOLINT(performance-move-const-arg)
    }
    ASSERT_EQ(
        ::testing::internal::GetCapturedStdout(),
        "myalgo_t: Hello from default implementation!\nWaving from 'then' with value Printable(value=42).\n");
}

//! @test Check that @ref iswd::domain customization for @ref myalgo_namespace::myalgo kicks in and works as expected.
TEST(some_other_namespace, customized) {
    ::testing::internal::CaptureStdout();
    {
        auto work = ::stdexec::schedule(iswd::inline_scheduler{})
                  | ::stdexec::let_value([]() noexcept { return ::stdexec::just(Printable{.value = 666}); })
                  | myalgo_namespace::myalgo("Hello from custom implementation!")
                  | ::tests::stdexec::check_scheduler<iswd::inline_scheduler>()
                  | ::stdexec::then(
                        [](const auto& value) { std::cout << "Waving from 'then' with value " << value << ".\n"; });
        ::stdexec::sync_wait(std::move(work)); // NOLINT(performance-move-const-arg)
    }
    ASSERT_EQ(
        ::testing::internal::GetCapturedStdout(),
        "MY BIG FAT CUSTOM PREFIX: Hello from custom implementation!\nWaving from 'then' with value "
        "Printable(value=666).\n");
}

} // namespace tests::stdexec::adaptors
