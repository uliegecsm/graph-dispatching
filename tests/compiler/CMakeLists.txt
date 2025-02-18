include_guard(GLOBAL)

include(CheckCXXSourceCompiles)

set(CODE_CPP_LIB_SENDERS "
    #if !__has_cpp_attribute(__cpp_lib_senders)
        #error \"This compiler does not support it.\"
    #endif

    int main() {}"
)

check_cxx_source_compiles("${CODE_CPP_LIB_SENDERS}" HAVE_CPP_LIB_SENDERS)

if(HAVE_CPP_LIB_SENDERS)
    message(FATAL_ERROR "We assumed your compiler should not support 'std::execution'.")
endif()
