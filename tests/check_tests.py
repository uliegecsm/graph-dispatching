import argparse
import logging
import os
import re
import subprocess

import typeguard

@typeguard.typechecked
def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--preset', type = str, required = True)

    return parser.parse_args()

@typeguard.typechecked
def expected_number_of_tests(*, preset : str) -> int:
    """
    Takes into account:
        - `CMake` preset
        - multi-GPU
    """
    count = None

    match preset:
        case 'clang-HPX' | 'clang-OpenMP' | 'gcc-OpenMP':
            count = 58
        case 'rocm-HIP':
            count = 65
        case 'clang-HPX-Cuda' | 'clang-Cuda':
            count = 70
        case 'gcc-Cuda':
            count = 37
        case _:
            raise ValueError(f'unsupported preset \'{preset}\'')

    if 'GRAPH_DISPATCHING_ENABLE_MULTIGPU' in os.environ and \
        os.environ['GRAPH_DISPATCHING_ENABLE_MULTIGPU'] == 'ON':
        count += 1

    return count

@typeguard.typechecked
def main(*, preset : str) -> None:
    """
    Check how many tests there are.
    """
    output = subprocess.check_output(['ctest', f'--preset={preset}', '-N']).decode()
    if (matched := re.search(pattern = r'Total Tests: ([0-9]+)', string = output)) is not None:
        matched = int(matched.group(1))
        expt    = expected_number_of_tests(preset = preset)
        logging.info(f'Checking that the number of tests is {expt}.')
        if expt != matched:
            raise RuntimeError(f'unexpected number of tests (expecting {expt} but got {matched})')
    else:
        raise RuntimeError(f'Could not find the number of tests for preset \'{preset}\'.')

if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO)

    args = parse_args()

    logging.info(f'Received arguments: {args}')

    main(**vars(args))
