import dataclasses
import enum
import json
import logging
import pathlib
import re
import subprocess
import typing

import matplotlib.pyplot
import matplot2tikz
import numpy
import pandas
import typeguard

from benchmarks.base import BenchmarkBase, parse_args
from benchmarks.common import asymptotic_speedup, n_be, step_like_plot, S_n

class CGFlavor(enum.StrEnum):
    """
    CG solver flavor.
    """
    GRAPH           = 'graph'
    GRAPH_WITH_HOST = 'graph_with_host'
    QUEUE           = 'queue'

@dataclasses.dataclass(frozen=True)
class Parameters:
    flavor: CGFlavor
    num_rows: int

class CGBenchmark(BenchmarkBase):

    @typeguard.typechecked
    def __init__(self, target: pathlib.Path) -> None:
        super().__init__(target=target)
        self.results: typing.Final[pathlib.Path] = pathlib.Path(f'{self.target}.json')

    @typeguard.typechecked
    def params(self, *, name : str) -> Parameters:
        """
        Retrieve benchmark parameters from its name.
        """
        pattern = rf'CGBenchmark/({"|".join(CGFlavor)})/num_rows:([0-9]+)/manual_time'

        self.assertRegex(name, pattern)

        match = re.match(pattern, name)

        return Parameters(flavor=CGFlavor(match.group(1)), num_rows=int(match.group(2)))

    @typeguard.typechecked
    def run(self, *, args : typing.List[str]) -> None:
        """
        Run the benchmark.
        """
        cmd = [
            self.target,
            '--benchmark_out=' + str(self.results),
            '--benchmark_out_format=json',
            '--benchmark_min_time=1x',
            '--benchmark_enable_random_interleaving=true',
            '--benchmark_min_warmup_time=0',
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
            benchmark_results = json.load(fin)

        # Regions of the graph implementation.
        FIELDS = ('graph definition', 'graph instantiation', 'graph submit 0', 'graph submit r')

        # Collect data for each case.
        data = {}
        data[CGFlavor.QUEUE] = {
            'num_rows': [],
            'real_time': [],
            'num_iters': [],
            'loop': [],
        }
        for flavor in (CGFlavor.GRAPH, CGFlavor.GRAPH_WITH_HOST):
            data[flavor] = {x: [] for x in FIELDS}
            data[flavor]['num_rows'] = []
            data[flavor]['real_time'] = []
            data[flavor]['num_iters'] = []

        time_unit = None

        for bench_case in benchmark_results['benchmarks']:
            logging.info(f'Analysing results of the benchmark case {bench_case["name"]} (it ran {bench_case["iterations"]} times, average real time {bench_case["real_time"]} {bench_case["time_unit"]}).')

            if time_unit is None:
                time_unit = bench_case['time_unit']
            else:
                self.assertEqual(time_unit, bench_case['time_unit'])

            params = self.params(name = bench_case['name'])

            data[params.flavor]['num_rows'].append(params.num_rows)
            data[params.flavor]['real_time' ].append(bench_case['real_time'])
            data[params.flavor]['num_iters'].append(bench_case['num_iters'])

            match params.flavor:
                case CGFlavor.GRAPH | CGFlavor.GRAPH_WITH_HOST:
                    for field in FIELDS:
                        self.assertGreater(bench_case[field], 0)
                        data[params.flavor][field].append(bench_case[field])

                case CGFlavor.QUEUE:
                    self.assertGreater(bench_case['loop'], 0.)
                    data[params.flavor]['loop'].append(bench_case['loop'])

                case _:
                    raise ValueError(params.flavor)

        # Transform to Pandas dataframes.
        data = {flavor: pandas.DataFrame(data[flavor]) for flavor in CGFlavor}

        # Sort according to increasing number of rows.
        for flavor in CGFlavor:
            data[flavor] = data[flavor].sort_values(by=['num_rows']).reset_index(drop=True)

        # Assert the available number of rows match
        num_rows = data[CGFlavor.QUEUE]['num_rows'].unique()
        numpy.testing.assert_array_equal(num_rows, data[CGFlavor.GRAPH]['num_rows'].unique())
        numpy.testing.assert_array_equal(num_rows, data[CGFlavor.GRAPH_WITH_HOST]['num_rows'].unique())

        # Normalize the 'queue' loop time w.r.t. the number of iterations.
        data[CGFlavor.QUEUE]['T_E'] = data[CGFlavor.QUEUE]['loop'] / data[CGFlavor.QUEUE]['num_iters']

        # Compute T_G_s0, T_G_sr, asymptotic speedup, n_be and S_n.
        for flavor in (CGFlavor.GRAPH, CGFlavor.GRAPH_WITH_HOST):
            data[flavor]['T_G_s0'] = data[flavor]['graph submit 0']
            data[flavor]['T_G_sr'] = data[flavor]['graph submit r'] / (data[flavor]['num_iters'] - 1)
            data[flavor]['S'] = asymptotic_speedup(T_E=data[CGFlavor.QUEUE]['T_E'], T_G_sr=data[flavor]['T_G_sr'])
            data[flavor]['n_be'] = n_be(
                T_E=data[CGFlavor.QUEUE]['T_E'],
                T_G_d=data[flavor]['graph definition'],
                T_G_i=data[flavor]['graph instantiation'],
                T_G_s0=data[flavor]['T_G_s0'],
                T_G_sr=data[flavor]['T_G_sr'],
            )
            data[flavor]['S_n'] = S_n(
                n_T_E=data[CGFlavor.QUEUE]['loop'],
                T_G_d=data[flavor]['graph definition'],
                T_G_i=data[flavor]['graph instantiation'],
                T_G_s0=data[flavor]['T_G_s0'],
                n_m1_T_G_sr=data[flavor]['graph submit r'],
            )
            data[flavor]['n_be_integer'] = data[flavor]['n_be'].apply(numpy.ceil).astype(int)

        # Debug logging.
        for flavor in CGFlavor:
            logging.info(f'Collected data for flavor {flavor}:\n{data[flavor]}')

        # Create the figure (one per graph flavor).
        for flavor in (CGFlavor.GRAPH, CGFlavor.GRAPH_WITH_HOST):
            fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1)

            # Plot the speedup after n submissions (S_n) and the asymptotic speedup (S) (left axis).
            COLOR_S = 'tab:red'
            ax_S = ax

            ax_S_min = numpy.floor(data[flavor]['S'].min() * 2) / 2
            ax_S_max = numpy.ceil (data[flavor]['S'].max() * 2) / 2

            ax_S.plot(num_rows, data[flavor]['S'],   linestyle='-', marker='o', color=COLOR_S)
            ax_S.set_ylabel('$S$ [-]', color=COLOR_S)
            ax_S.set_xlabel('Number of rows [-]')
            ax_S.tick_params(axis='y', labelcolor=COLOR_S)
            ax_S.set_xlim(min(num_rows), max(num_rows))
            ax_S.set_ylim(ax_S_min, ax_S_max)
            ax_S.set_xscale('log')
            ax_S.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda x, _: f'{x:.1f}'.rstrip('0').rstrip('.')
            ))

            # Plot 'n_be' where it makes sense (i.e. where S >= 1) (right axis).
            COLOR_N_BE = 'tab:blue'

            if (mask := data[flavor]['n_be_integer'] > 0).any():
                ax_n_be = ax_S.twinx()
                step_like_plot(
                    ax=ax_n_be,
                    x=data[flavor]['num_rows'].values,
                    y=data[flavor]['n_be_integer'].values,
                    predicate=mask,
                    color=COLOR_N_BE,
                )
                ax_n_be.set_ylabel(r'$n_{\text{be}}$ [-]', color=COLOR_N_BE)
                ax_n_be.tick_params(axis='y', labelcolor=COLOR_N_BE)

                n_be_integer_max = int(data[flavor]['n_be_integer'][mask].max())
                n_be_num_ticks = min(5, n_be_integer_max + 1)
                n_be_ticks = numpy.unique(numpy.linspace(0, n_be_integer_max, n_be_num_ticks).round().astype(int))
                n_be_min, n_be_max = 0., max(n_be_ticks) + 0.5
                ax_n_be.set_ylim(n_be_min, n_be_max)
                ax_n_be.set_yticks(n_be_ticks)

                # Align S ticks to n_be ticks so the horizontal grid looks clean.
                fractions = (n_be_ticks - n_be_min) / (n_be_max - n_be_min)
                ax_S.set_yticks(ax_S_min + fractions * (ax_S_max - ax_S_min))

            ax_S.yaxis.grid(True, linestyle='--', alpha=0.5)

            # Save the figure.
            for ext in ('svg', 'eps'):
                saved_to = self.target.with_suffix(f'.{flavor}.S_and_n_be.{ext}')
                logging.info(f'Saving plot of n_be to {saved_to}.')
                fig.savefig(saved_to, bbox_inches=0, transparent=False)

            matplot2tikz.save(figure=fig, filepath=self.target.with_suffix(f'.{flavor}.S_and_n_be.tex'))

if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO)

    args = parse_args()

    logging.info(f"Received arguments: {args}")

    runner = CGBenchmark(target = args.target)

    runner.run(args = args.target_args)

    runner.analyse()
