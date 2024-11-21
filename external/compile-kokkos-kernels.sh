set -ex

KOKKOSKERNELS_PRESET=$1
KOKKOSKERNELS_SOURCES=$PWD/external/kokkos-kernels
KOKKOSKERNELS_INSTALL=$PWD/kokkos-kernels-compiled

echo "> Compiling Kokkos Kernels with preset ${KOKKOSKERNELS_PRESET} from ${KOKKOSKERNELS_SOURCES} and installing to ${KOKKOSKERNELS_INSTALL}."

# Make sure to use our compiled Kokkos.
export Kokkos_ROOT=$PWD/kokkos-compiled/$KOKKOSKERNELS_PRESET

# Create install directory.
mkdir -p $KOKKOSKERNELS_INSTALL

# Compile sources and install.
cd $KOKKOSKERNELS_SOURCES

cmake -S . -B build-kokkos-kernels-${KOKKOSKERNELS_PRESET} -Wno-error=dev \
    -DKokkosKernels_ENABLED_COMPONENTS=SPARSE \
    -DCMAKE_INSTALL_PREFIX=${KOKKOSKERNELS_INSTALL}/${KOKKOSKERNELS_PRESET}

cmake --build build-kokkos-kernels-${KOKKOSKERNELS_PRESET} --target=install -j4
