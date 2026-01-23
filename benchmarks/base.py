import argparse
import pathlib
import typing
import unittest

import typeguard

class BenchmarkBase(unittest.TestCase):
    @typeguard.typechecked
    def __init__(self, target : pathlib.Path) -> None:
        super().__init__()
        self.target: typing.Final[pathlib.Path] = target.resolve()

@typeguard.typechecked
def parse_args() -> argparse.Namespace:
    """
    Parse CLI args.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--target', type=pathlib.Path, required=True)

    parser.add_argument(dest='target_args' , help="Arguments that will be passed to the 'target'.", nargs='*')

    return parser.parse_args()
