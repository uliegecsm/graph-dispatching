#include "benchmark/benchmark.h"

#include "Kokkos_Core.hpp"

int main(int argc, char** argv)
{
    ::benchmark::MaybeReenterWithoutASLR(argc, argv);

    //! Instruct the tools to avoid global fences if possible.
    Kokkos::Tools::Experimental::set_request_tool_settings_callback(
        [](const uint32_t, Kokkos::Tools::Experimental::ToolSettings* settings) {
            settings->requires_global_fencing = false;
        });

    Kokkos::initialize(argc, argv);

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    Kokkos::finalize();

    return EXIT_SUCCESS;
}
