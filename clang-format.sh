#!/bin/bash
set -ex

CLANG_FORMAT_EXECUTABLE=clang-format-21

PATTERNS=(
    'tests/kokkos_ext/execution_space/test_scheduler.cpp'
    'tests/utils/*'
)

for pattern in "${PATTERNS[@]}"; do
    if ! git ls-files "$pattern" | grep -q .; then
        echo "ERROR: Pattern '$pattern' did not match."
        exit 1
    fi
done

git ls-files "${PATTERNS[@]}" | xargs ${CLANG_FORMAT_EXECUTABLE} --dry-run --Werror
