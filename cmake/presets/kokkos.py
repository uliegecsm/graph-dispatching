import argparse
import logging
import pathlib

import typeguard

SEARCH_STR = 'REPL_STR_KOKKOS_ARCH_ENABLED'

@typeguard.typechecked
def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-file',  type = pathlib.Path, required = True)
    parser.add_argument('--output-file', type = pathlib.Path, required = False)

    parser.add_argument('--kokkos-archs', type = str, required = False)

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.input_file

    return args

@typeguard.typechecked
def main(*, input_file : pathlib.Path, output_file : pathlib.Path, kokkos_archs : str) -> None:
    """
    Replace with the enabled `Kokkos` architectures.
    """
    archs = kokkos_archs.split(',')

    replace_str = ''

    if len(archs) > 0:
        for arch in archs:
            replace_str += f'"Kokkos_ARCH_{arch}" : "ON",'
        logging.info(f'Enabling Kokkos architectures {archs} in {output_file} (from {input_file}) with:\n\t{replace_str}')
    else:
        logging.warning(f'No Kokkos architecture enabled.')

    output_file.write_text(input_file.read_text().replace(SEARCH_STR, replace_str))

if __name__ == "__main__":

    logging.basicConfig(level = logging.INFO)

    args = parse_args()

    logging.info(f'Received arguments: {args}')

    main(**vars(args))
