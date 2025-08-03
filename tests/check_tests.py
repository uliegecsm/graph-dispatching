import argparse
import logging
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
    match preset:
        case 'gcc-Cuda':
            return 31
        case _:
            raise ValueError(f'unsupported preset \'{preset}\'')

@typeguard.typechecked
def main(*, preset : str) -> None:
    """
    Check how many tests there are.
    """
    output = subprocess.check_output(['ctest', f'--preset={preset}', '-N']).decode()
    if (matched := re.search(pattern = r'Total Tests: ([0-9]+)', string = output)) is not None:
        matched = int(matched.group(1))
        if (expt := expected_number_of_tests(preset = preset)) != matched:
            raise RuntimeError(f'unexpected number of tests (expecting {expt} but got {matched})')
    else:
        raise RuntimeError(f'Could not find the number of tests for preset \'{preset}\'.')

if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO)

    args = parse_args()

    logging.info(f'Received arguments: {args}')

    main(**vars(args))
