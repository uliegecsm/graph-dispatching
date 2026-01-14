#!/bin/bash
set -ex

CLANG_FORMAT_EXECUTABLE=clang-format-21

if [ "$#" -eq 0 ]; then
  CLANG_FORMAT_ARGS=(--dry-run --Werror)
else
  CLANG_FORMAT_ARGS=("$@")
fi

PATTERNS=(
    'kokkos_ext/impl/execution_space/sync_wait.hpp'
    'kokkos_ext/impl/GraphContext_fwd.hpp'
    'kokkos_ext/impl/GraphContext.hpp'
    'kokkos_ext/impl/sync_wait.hpp'
    'tests/kokkos_ext/execution_space/test_scheduler.cpp'
    'tests/kokkos_ext/graph/*'
    'tests/utils/*'
)

for pattern in "${PATTERNS[@]}"; do
    if ! git ls-files "$pattern" | grep -q .; then
        echo "ERROR: Pattern '$pattern' did not match."
        exit 1
    fi
done

git ls-files "${PATTERNS[@]}" | xargs ${CLANG_FORMAT_EXECUTABLE} "${CLANG_FORMAT_ARGS[@]}"
