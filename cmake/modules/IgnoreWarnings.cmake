# This file contains setup w.r.t. compiler flags related to warnings.
include_guard(GLOBAL)

if(
    CMAKE_CXX_COMPILER_ID MATCHES "Clang"
    OR
    CMAKE_CXX_COMPILER_ID MATCHES "GNU"
)
    # For GNU GCC, warnings are listed at https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html.
    add_compile_options(
        -Wall
        -Wextra
        -Werror=dangling-else
        -Werror=shadow
        -Werror=switch-default
        -Werror=suggest-override
        -Werror=overloaded-virtual
        -Werror
        # Prevent cross execution space call on CUDA. More generally treat any warning as an error.
        "$<$<IN_LIST:CUDA,${Kokkos_DEVICES}>:--Werror=cross-execution-space-call,all-warnings>"
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # For Clang, diagnostic flags are listed at https://clang.llvm.org/docs/DiagnosticsReference.html.
        add_compile_options(
            -Werror=unused-private-field
            -Werror=unused-lambda-capture
            -Werror=unused-member-function
            -Werror=delete-non-virtual-dtor
        )
    endif()

else()
    message(WARNING "You compiler ${CMAKE_CXX_COMPILER_ID} is not supported for additional warning options.")
endif()
