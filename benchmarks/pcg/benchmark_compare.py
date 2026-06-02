import dataclasses
import enum
import itertools
import json
import logging
import pathlib
import re
import subprocess
import typing

import matplotlib.pyplot
import numpy
import numpy.typing
import typeguard

from benchmarks.base import BenchmarkBase, parse_args

class PCGFlavor(enum.StrEnum):
    """
    PCG solver flavor.
    """
    GRAPH = 'graph'
    QUEUE = 'queue'

@dataclasses.dataclass(eq = True, frozen = True)
class Parameters:
    nrows : int
    niters : int
    nsweeps : int

@dataclasses.dataclass(frozen = True)
class Results:
    nreps : int
    mean : float
    timings : numpy.typing.NDArray[float] = None

class PCGBenchmark(BenchmarkBase):

    @typeguard.typechecked
    def __init__(self, target: pathlib.Path) -> None:
        super().__init__(target=target)
        self.results = self.target.with_suffix('.json')

    @typeguard.typechecked
    def params(self, *, name : str) -> typing.Tuple[PCGFlavor, Parameters]:
        """
        Retrieve benchmark case parameters from its name.
        """
        pattern = rf'PCGBenchmark/({"|".join(list(PCGFlavor))})/nrows:([0-9]+)/niters:([0-9]+)/nsweeps:([0-9]+)/manual_time'

        self.assertRegex(name, pattern)

        match = re.match(pattern, name)

        return PCGFlavor(match.group(1)), Parameters(
            nrows = int(match.group(2)),
            niters = int(match.group(3)),
            nsweeps = int(match.group(4)),
        )

    @typeguard.typechecked
    def run(self, *, args : typing.List[str]) -> None:
        """
        Run the benchmark.
        """
        cmd = [
            self.target,
            '--benchmark_out=' + str(self.results),
            '--benchmark_out_format=json',
            '--benchmark_min_time=2x',
            '--benchmark_enable_random_interleaving=true',
            *args,
        ]

        logging.info(f'Running benchmark with {cmd}.')

        subprocess.check_call(cmd)

    @typeguard.typechecked
    def analyse(self) -> None:
        """
        Analyse the results.
        """
        # Load benchmark results.
        logging.info(f'Loading results from {self.results}.')
        with open(self.results, 'r') as fin:
            self.benchmark_results = json.load(fin)

        # Collect data for each case.
        time_unit = None

        data : typing.Dict[PCGFlavor, typing.Dict[Parameters, Results]] = {x : {} for x in PCGFlavor}

        for bench_case in self.benchmark_results['benchmarks']:
            logging.info(f'Analysing results of the benchmark case {bench_case['name']} (it ran {bench_case["iterations"]} times, average real time {bench_case["real_time"]} {bench_case["time_unit"]}).')

            if time_unit is None:
                time_unit = bench_case['time_unit']
            else:
                self.assertEqual(time_unit, bench_case['time_unit'])

            flavor, params = self.params(name = bench_case['name'])

            self.assertTrue(numpy.allclose(bench_case['real_time'], bench_case['mean']))

            bench_output = pathlib.Path(self.target.parent) / bench_case['name'].removesuffix('manual_time').replace(':', '_') / 'timings.bin'

            self.assertTrue(bench_output.is_file(), msg = bench_output)

            timings = numpy.fromfile(file = bench_output, dtype = float, sep = '')

            self.assertEqual(timings.shape, (bench_case['nreps'],))

            data[flavor][params] = Results(
                nreps   = bench_case['nreps'],
                mean    = bench_case['mean'],
                timings = timings,
            )

        # Number of sweeps (unique, sorted).
        nsweeps_sorted_set = sorted(set([k.nsweeps for k in data[PCGFlavor.GRAPH].keys()]))
        self.assertEqual(    sorted(set([k.nsweeps for k in data[PCGFlavor.QUEUE].keys()])), nsweeps_sorted_set)

        logging.info(f'List of sweeps: {nsweeps_sorted_set}.')

        self.assertEqual(len(nsweeps_sorted_set), 3)

        # Number of rows (unique, sorted).
        nrows_sorted_set = sorted(set([k.nrows for k in data[PCGFlavor.GRAPH].keys()]))
        self.assertEqual(  sorted(set([k.nrows for k in data[PCGFlavor.QUEUE].keys()])), nrows_sorted_set)

        logging.info(f'List of rows: {nrows_sorted_set}.')

        nrows_sorted_set = numpy.asarray(nrows_sorted_set)

        # The baseline is the 'queue' flavor.
        BASELINE = PCGFlavor.QUEUE

        ratio_full = {x : [] for x in nsweeps_sorted_set}

        for params, results in data[PCGFlavor.GRAPH].items():
            ratio = results.mean / data[BASELINE][params].mean
            ratio_full[params.nsweeps].append((params.nrows, ratio))

        for nsweeps in nsweeps_sorted_set:
            ratio_full[nsweeps] = sorted(ratio_full[nsweeps], key = lambda x : x[0])

        # Create figure that shows both real time and ratio.
        _, axes = matplotlib.pyplot.subplots(nrows = 2, ncols = 1, figsize = (10, 7))
        ax_time  = axes[0]
        ax_ratio = axes[1]

        COLORS: typing.Final[tuple[str, str]] = ('r', 'b', 'black')

        for insweeps, nsweeps in enumerate(nsweeps_sorted_set):
            ax_ratio.plot(
                [x[0] for x in ratio_full[nsweeps]],
                [x[1] for x in ratio_full[nsweeps]],
                label = nsweeps,
                color = COLORS[insweeps],
            )

        for flavor, (insweeps, nsweeps) in itertools.product(PCGFlavor, enumerate(nsweeps_sorted_set)):

            match flavor:
                case PCGFlavor.GRAPH:
                    linestyle = '-'
                case _:
                    linestyle = ':'

            color = COLORS[insweeps]

            @typeguard.typechecked
            def collect(d : dict, key : typing.Callable, value : typing.Callable, condition : typing.Callable) -> list:
                collected = [(key(k), value(v)) for k, v in d.items() if condition(k)]
                collected = [x[1] for x in sorted(collected, key = lambda x : x[0])]
                return collected

            means   = collect(d = data[flavor], key = lambda x: x.nrows, value = lambda x: x.mean,    condition = lambda x: x.nsweeps == nsweeps)
            timings = collect(d = data[flavor], key = lambda x: x.nrows, value = lambda x: x.timings, condition = lambda x: x.nsweeps == nsweeps)
            percent = [
                numpy.percentile(a = x, q = [2.5, 97.5])
                for x in timings
            ]

            label = f'{flavor} - {nsweeps}'

            ax_time.plot        (nrows_sorted_set, means, label = label, linestyle = linestyle, color = color)
            ax_time.fill_between(nrows_sorted_set, [x[0] for x in percent], [x[1] for x in percent], alpha = .5, linewidth = 0, color = 'grey')

        ax_ratio.legend()
        ax_time .legend()

        ax_time .set_xscale('log')
        ax_ratio.set_xscale('log')

        ax_ratio.set_xlabel('size [-]')
        ax_ratio.set_ylabel('overall ratio [-]')

        ax_time.set_yscale('log')
        ax_time.set_ylabel(f'time to solution [{time_unit}]')

        # Plot the 'one' ratio.
        ax_ratio.axhline(y = 1., color = 'black', linestyle = '-')

        # Enable grid.
        ax_ratio.grid(True)
        ax_time .grid(True)

        # Save figure.
        for ext in ('svg', 'eps', 'png'):
            output = self.results.with_suffix('.' + ext)
            logging.info(f'Saving figure to {output}.')
            matplotlib.pyplot.savefig(output, bbox_inches=0, transparent=False)

if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO)

    args = parse_args()

    logging.info(f"Received arguments: {args}")

    runner = PCGBenchmark(target = args.target)

    runner.run(args = args.target_args)

    runner.analyse()
