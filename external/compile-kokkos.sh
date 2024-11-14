set -ex

KOKKOS_PRESET=$1
KOKKOS_SOURCES=$PWD/external/kokkos
KOKKOS_INSTALL=$PWD/kokkos-compiled

echo "> Compiling Kokkos with preset ${KOKKOS_PRESET} from ${KOKKOS_SOURCES} and installing to ${KOKKOS_INSTALL}."

# Retrieve CMake presets.
cp $PWD/cmake/presets/kokkos.json ${KOKKOS_SOURCES}/CMakePresets.json

# Create install directory.
mkdir -p $KOKKOS_INSTALL

# Compile sources and install.
cd $KOKKOS_SOURCES

cmake -S . --preset=${KOKKOS_PRESET} -Wno-error=dev \
    -DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL}/${KOKKOS_PRESET}

cmake --build --preset=${KOKKOS_PRESET} --target=install -j4
