set -ex

WORKSPACE=$PWD

mkdir -p $WORKSPACE/kokkos-compiled

cd $WORKSPACE/external/kokkos

cp $WORKSPACE/cmake/presets/kokkos.json $WORKSPACE/external/kokkos/CMakePresets.json

KOKKOS_PRESET=clang-Cuda

cmake -S . --preset=${KOKKOS_PRESET} \
    -DCMAKE_INSTALL_PREFIX=$WORKSPACE/kokkos-compiled/${KOKKOS_PRESET}

cmake --build --preset=${KOKKOS_PRESET} -j4

cmake --build --preset=${KOKKOS_PRESET} --target=install