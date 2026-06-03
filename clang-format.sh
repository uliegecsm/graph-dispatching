#!/bin/bash
set -ex

CLANG_FORMAT_EXECUTABLE=clang-format-21

if [ "$#" -eq 0 ]; then
  CLANG_FORMAT_ARGS=(--dry-run --Werror)
else
  CLANG_FORMAT_ARGS=("$@")
fi

PATTERNS=(
    'algorithms/newton/*.hpp'
    'apps/**.hpp'
    'benchmarks/cg/benchmark_compare.cpp'
    'benchmarks/pcg/benchmark_compare.cpp'
    'benchmarks/newton/benchmark_compare.cpp'
    'examples/**.cpp'
    'examples/**.hpp'
    'kokkos_ext/impl/execution_space/*.hpp'
    'kokkos_ext/impl/graph/*.hpp'
    'kokkos_ext/impl/GraphContext_fwd.hpp'
    'kokkos_ext/impl/GraphContext.hpp'
    'kokkos_ext/impl/bulk.hpp'
    'kokkos_ext/impl/completion_signatures.hpp'
    'kokkos_ext/impl/env.hpp'
    'kokkos_ext/impl/parallel_for.hpp'
    'kokkos_ext/impl/sync_wait.hpp'
    'tests/exec/**.cpp'
    'tests/kokkos/**.cpp'
    'tests/kokkos_ext/*.hpp'
    'tests/kokkos_ext/test_env.cpp'
    'tests/kokkos_ext/execution_space/*.cpp'
    'tests/kokkos_ext/graph/*.cpp'
    'tests/kokkos_ext/graph/*.hpp'
    'tests/main.cpp'
    'tests/newton/*.cpp'
    'tests/newton/*.hpp'
    'tests/nvexec/**.cpp'
    'tests/stdexec/**.cpp'
    'tests/stdexec/**.hpp'
    'tests/utils/*'
    'tests/CallbackMatchers.hpp'
    'tests/Functors.hpp'
)

for pattern in "${PATTERNS[@]}"; do
    if ! git ls-files "$pattern" | grep -q .; then
        echo "ERROR: Pattern '$pattern' did not match."
        exit 1
    fi
done

git ls-files "${PATTERNS[@]}" | xargs ${CLANG_FORMAT_EXECUTABLE} "${CLANG_FORMAT_ARGS[@]}"
