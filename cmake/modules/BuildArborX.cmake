include(FetchContent)

FetchContent_Declare(
    ArborX
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/ArborX
    EXCLUDE_FROM_ALL
)

set(ARBORX_ENABLE_MPI OFF)

FetchContent_MakeAvailable(ArborX)

if(NOT TARGET ArborX::ArborX)
    message(FATAL_ERROR "ArborX should define a ArborX::ArborX target.")
endif()
