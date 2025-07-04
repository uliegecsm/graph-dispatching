import argparse
import logging
import pathlib
import re

import typeguard

@typeguard.typechecked
def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--runs-on', type = str, required = True)
    parser.add_argument('--output',  type = pathlib.Path, required = True)

    return parser.parse_args()

@typeguard.typechecked
def main(*, runs_on : str, output : pathlib.Path) -> None:
    """
    Detect how many GPUs are requested in `runs_on`.
    """
    m = set(re.findall(pattern = r'\'(gpu\:[0-9]+)\'', string = runs_on))

    logging.info(f'Detected GPU labels: {m}.')

    with output.open(mode = 'a+') as fout:
        fout.write(f'GRAPH_DISPATCHING_ENABLE_MULTIGPU={"ON" if len(m) > 1 else "OFF"}\n')
        fout.write(f'GRAPH_DISPATCHING_GPU_COUNT={len(m)}\n')

if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO)

    args = parse_args()

    logging.info(f'Received arguments: {args}')

    main(**vars(args))
