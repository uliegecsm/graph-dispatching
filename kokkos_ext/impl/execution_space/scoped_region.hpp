#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SCOPED_REGION_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SCOPED_REGION_HPP

#include <format>

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "impl/Kokkos_Profiling.hpp"

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"

/**
 * @file
 *
 * This file provides the implementation of algorithms related to @c Kokkos::Profiling push/pop regions.
 *
 * References:
 *  - https://github.com/NVIDIA/stdexec/blob/f7308ea245b896a76c6fd9a58a097ae23579e489/include/nvexec/nvtx.cuh
 */

namespace Kokkos::Experimental::details::execution_space
{
//! Kind of region action.
enum class Kind : std::uint8_t
{
    PUSH,
    POP
};

template <Kind kind, stdexec::receiver Rcvr, stdexec::scheduler Schd> requires stdexec::__is_instance_of_<Schd, ExecutionSpaceScheduler>
struct RegionReceiver
{
    using receiver_concept = stdexec::receiver_t;

    Rcvr rcvr;
    std::string name;
    Schd schd;

    template <typename Tag, typename... Args>
    void complete(Tag tag, Args&&... args) && noexcept
    {
        std::move(schd).env.exec.fence(std::format("{}: {}", Kokkos::Impl::TypeInfo<decltype(schd.env.exec)>::name(), kind == Kind::PUSH ? "push" : "pop"));

        if constexpr (kind == Kind::PUSH) {
            Kokkos::Profiling::pushRegion(name);
        } else {
            Kokkos::Profiling::popRegion();
        }

        std::invoke(tag, std::move(rcvr), std::forward<Args>(args)...);
    }

    void set_value() && noexcept {
        std::move(*this).complete(stdexec::set_value);
    }

    template <typename Error>
    void set_error(Error&& err) && noexcept {
        std::move(*this).complete(stdexec::set_error, std::forward<Error>(err));
    }

    decltype(auto) get_env() const noexcept { return stdexec::get_env(rcvr); }
};

template <Kind kind, stdexec::sender Sndr>
struct RegionSender
{
    using sender_concept = stdexec::sender_t;

    template <typename Self, typename... Env>
    using _completion_signatures = stdexec::transform_completion_signatures<
        stdexec::completion_signatures_of_t<stdexec::__copy_cvref_t<Self, Sndr>, Env...>
    >;

    //! As required by https://github.com/NVIDIA/stdexec/blob/3363435259b7ffae43d3f2e5f6b7a7b36d7cd7d3/include/stdexec/__detail/__diagnostics.hpp#L266-L310.
    template <class... Env>
    [[nodiscard]] constexpr auto
    get_completion_signatures(Env&&...) -> _completion_signatures<RegionSender, Env...> { return {}; } // NOLINT(cppcoreguidelines-missing-std-forward)

    template <stdexec::receiver Rcvr>
    stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>)
    {
        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr));

        using recv_t = RegionReceiver<kind, std::remove_cvref_t<Rcvr>, std::remove_cvref_t<decltype(schd)>>;

        return stdexec::connect(
            std::move(sndr),
            recv_t{.rcvr = std::forward<Rcvr>(rcvr), .name = std::move(name), .schd = std::move(schd)}
        );
    }

    Sndr sndr;
    std::string name {};

    decltype(auto) get_env() const noexcept { return stdexec::get_env(sndr); }
};

struct Push
{
    template <stdexec::sender Sndr, typename T>
    auto operator()(Sndr&& sndr, T&& name) const noexcept -> RegionSender<Kind::PUSH, Sndr> {
        return RegionSender<Kind::PUSH, Sndr>{.sndr = std::forward<Sndr&&>(sndr), .name = std::forward<T>(name)};
    }

    template <typename T>
    auto operator()(T&& name) const noexcept {
        return stdexec::__closure{*this, std::forward<T>(name)};
    }
};

struct Pop
{
    template <stdexec::sender Sndr>
    auto operator()(Sndr&& sndr) const noexcept -> RegionSender<Kind::POP, Sndr> {
        return RegionSender<Kind::POP, Sndr>{.sndr = std::forward<Sndr>(sndr)};
    }

    auto operator()() const noexcept {
        return stdexec::__closure{*this};
    }
};

//! Helper for @c Kokkos::Profiling::scoped_region.
struct ScopedRegion
{
    template <stdexec::sender Sndr, typename T, stdexec::__sender_adaptor_closure Closure>
    auto operator()(Sndr&& sndr, T&& name, Closure&& closure) const noexcept {
        return std::forward<Sndr&&>(sndr) | Push{}(std::forward<T>(name)) | std::forward<Closure>(closure) | Pop{}();
    }

    template <typename T, stdexec::__sender_adaptor_closure Closure>
    auto operator()(T&& name, Closure&& closure) const noexcept {
        return stdexec::__closure{
            *this,
            std::forward<T>(name),
            std::forward<Closure>(closure)
        };
    }
};

} // namespace Kokkos::Experimental::details::execution_space

namespace Kokkos::Profiling
{
inline constexpr Kokkos::Experimental::details::execution_space::Push push{};
inline constexpr Kokkos::Experimental::details::execution_space::Pop pop{};
inline constexpr Kokkos::Experimental::details::execution_space::ScopedRegion scoped_region{};
} // namespace Kokkos::Profiling

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SCOPED_REGION_HPP
