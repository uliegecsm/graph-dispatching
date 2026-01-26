import argparse
import json
import logging
import pathlib
import subprocess

import typeguard

@typeguard.typechecked
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, required=True)
    parser.add_argument('--version-file', type=pathlib.Path, required=True)
    return parser.parse_args()

@typeguard.typechecked
def install_cuda_requirements(version: dict) -> None:
    logging.info('Installing requirements for the Cuda backend.')

    subprocess.check_call(('pip', 'install', 'reprospect==' + version['dependencies']['reprospect']['value']))

@typeguard.typechecked
def main(*, backend: str, version_file: pathlib.Path) -> None:
    backends = backend.split('-')
    logging.info(f'Installing requirements for backends {backends}.')

    with open(version_file, 'r') as f:
        version = json.load(f)

    for x in backends:
        match x:
            case 'Cuda':
                install_cuda_requirements(version=version)
            case _:
                logging.info(f'There is no requirement yet for backend {x}.')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(**vars(parse_args()))
