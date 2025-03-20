import argparse
import json
import logging
import pathlib
import typing

import typeguard

@typeguard.typechecked
def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-file',  type = pathlib.Path, required = True)
    parser.add_argument('--output-file', type = pathlib.Path, required = False)
    parser.add_argument('--preset',      type = str,          required = True)

    return parser.parse_args()

class JSONWithCommentsDecoder(json.JSONDecoder):
    """
    Ignore the C-like comment lines.
    """
    @typeguard.typechecked
    def decode(self, s: str) -> typing.Any:
        s = '\n'.join(l for l in s.split('\n') if not l.lstrip(' ').startswith('//'))
        return super().decode(s)

@typeguard.typechecked
def main(*, input_file : pathlib.Path, preset : str, output_file : typing.Optional[pathlib.Path] = None) -> None:
    """
    Add `preset` to `buildPresets` from `input_file` and dump the new presets in `output_file`.
    """
    with open(input_file, 'r') as infile:
        content = json.load(infile, cls = JSONWithCommentsDecoder)

    add = {
        "name"            : preset,
        "configurePreset" : preset,
        "inherits"        : "default",
    }

    content['buildPresets'].append(add)

    with open(output_file or input_file, 'w+') as outfile:
        json.dump(content, outfile, indent = 4)

if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO)

    args = parse_args()

    logging.info(f'Received arguments: {args}')

    main(**vars(args))
