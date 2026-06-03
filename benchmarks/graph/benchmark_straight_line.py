import dataclasses
import json
import logging
import pathlib
import re
import subprocess
import typing

import matplotlib.pyplot
import matplotlib.ticker
import numpy
import numpy.testing
import pandas
import typeguard

import matplot2tikz

from benchmarks.base import BenchmarkBase, parse_args
from benchmarks.common import asymptotic_speedup, n_be, step_like_plot
from benchmarks.graph.common import Method

@dataclasses.dataclass(frozen=True)
class Parameters:
    method: Method
    num_submits: int
    num_nodes: int

class StraightLineBenchmark(BenchmarkBase):

    @typeguard.typechecked
    def __init__(self, target: pathlib.Path) -> None:
        super().__init__(target=target)
        self.results = f'{self.target}.json'

    @typeguard.typechecked
    def params(self, *, name: str) -> Parameters:
        """
        Retrieve benchmark parameters from its name.
        """
        pattern = rf'StraightLineBenchmark/({"|".join(Method)})/num_submits:([0-9]+)/num_nodes:([0-9]+)'

        self.assertRegex(name, pattern)

        match = re.match(pattern, name)

        return Parameters(
            method=Method(match.group(1)),
            num_submits=int(match.group(2)),
            num_nodes=int(match.group(3)),
        )

    @typeguard.typechecked
    def run(self, *, args: typing.List[str]) -> None:
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

        # Collect data for each case.
        data = {}
        data[Method.EXEC] = {
            'num_submits': [],
            'num_nodes': [],
            'real_time': [],
        }
        data[Method.GRAPH] = {
            'num_submits': [],
            'num_nodes': [],
            'real_time': [],
            'create': [],
            'instantiate': [],
            'per_submit': [],
            'all_submits': [],
        }
        time_unit = None

        for bench_case in benchmark_results['benchmarks']:
            logging.info(f'Analysing results of the benchmark case {bench_case['name']} (it ran {bench_case["iterations"]} times, average real time {bench_case["real_time"]} {bench_case["time_unit"]}).')

            if time_unit is None:
                time_unit = bench_case['time_unit']
            else:
                self.assertEqual(time_unit, bench_case['time_unit'])

            params = self.params(name=bench_case['name'])

            logging.info(params)
            logging.info(bench_case)

            data[params.method]['num_submits'].append(params.num_submits)
            data[params.method]['num_nodes'].append(params.num_nodes)
            data[params.method]['real_time'].append(bench_case['real_time'])

            if params.method == Method.GRAPH:
                for field in ('create', 'instantiate', 'per_submit', 'all_submits'):
                    data[params.method][field].append(bench_case[field])

        # Transform to Pandas dataframes.
        data_exec = pandas.DataFrame(data[Method.EXEC])
        data_graph = pandas.DataFrame(data[Method.GRAPH])

        # Find out the available line lengths.
        exec_lengths = numpy.sort(data_exec['num_nodes'].unique())
        graph_lengths = numpy.sort(data_graph['num_nodes'].unique())
        numpy.testing.assert_array_equal(exec_lengths, graph_lengths)

        # Sort according to increasing line length.
        data_exec = data_exec.sort_values(by=['num_nodes']).reset_index(drop=True)
        data_graph = data_graph.sort_values(by=['num_nodes']).reset_index(drop=True)

        # Normalize the 'exec' real time w.r.t. the number of submissions.
        data_exec['T_E'] = data_exec['real_time'] / data_exec['num_submits']

        # Compute the asymptotic speedup.
        data_graph['S'] = asymptotic_speedup(T_E=data_exec['T_E'], T_G_sr=data_graph['per_submit'])

        # Compute the cost of the first submission.
        data_graph['T_G_s0'] = data_graph['all_submits'] - data_graph['num_submits'] * data_graph['per_submit']

        # Compute the number of submissions needed to compensate the graph definition and instantiation stages.
        data_graph['n_be'] = n_be(
            T_E=data_exec['T_E'],
            T_G_d=data_graph['create'],
            T_G_i=data_graph['instantiate'],
            T_G_s0=data_graph['T_G_s0'],
            T_G_sr=data_graph['per_submit'],
        )

        data_graph['n_be_integer'] = data_graph['n_be'].apply(numpy.ceil).astype(int)

        logging.info(f'Data for {Method.EXEC}:\n{data_exec}')
        logging.info(f'Data for {Method.GRAPH}:\n{data_graph}')

        # Filter out any result to is not for 10 submissions.
        data_graph = data_graph[data_graph['num_submits'] == 10]

        # Create the figure.
        fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1)

        # Plot the asymptotic speedup (left axis).
        COLOR_S = 'tab:red'
        ax_S = ax

        ax_S_min = numpy.floor(data_graph['S'].min() * 2) / 2
        ax_S_max = numpy.ceil(data_graph['S'].max() * 2) / 2

        ax_S.plot(data_graph['num_nodes'], data_graph['S'], marker='o', color=COLOR_S)
        ax_S.set_xlabel('Number of nodes [-]')
        ax_S.set_ylabel('$S$ [-]', color=COLOR_S)
        ax_S.tick_params(axis='y', labelcolor=COLOR_S)
        ax_S.set_xlim(min(exec_lengths), max(exec_lengths))
        ax_S.set_ylim(ax_S_min, ax_S_max)
        ax_S.set_xscale('log')
        ax_S.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, _: f'{x:.1f}'.rstrip('0').rstrip('.')
        ))

        # Plot 'n_be' where it makes sense (i.e. where S >= 1) (right axis).
        COLOR_N_BE = 'tab:blue'
        ax_n_be = ax_S.twinx()

        mask = data_graph['S'] >= 1

        step_like_plot(
            ax=ax_n_be,
            x=data_graph['num_nodes'].values,
            y=data_graph['n_be_integer'].values,
            predicate=mask,
            color=COLOR_N_BE,
        )
        ax_n_be.set_ylabel(r'$n_{\text{be}}$ [-]', color=COLOR_N_BE)
        ax_n_be.tick_params(axis='y', labelcolor=COLOR_N_BE)

        n_be_integer_max = int(data_graph['n_be_integer'][mask].max())
        n_be_num_ticks = min(5, n_be_integer_max + 1)
        n_be_ticks = numpy.unique(numpy.linspace(0, n_be_integer_max, n_be_num_ticks).round().astype(int))
        n_be_min, n_be_max = 0., max(n_be_ticks) + 0.5
        ax_n_be.set_ylim(n_be_min, n_be_max)
        ax_n_be.set_yticks(n_be_ticks)

        # Align S ticks to n_be ticks so the horizontal grid looks clean.
        fractions = (n_be_ticks - n_be_min) / (n_be_max - n_be_min)
        ax_S.set_yticks(ax_S_min + fractions * (ax_S_max - ax_S_min))

        ax_n_be.yaxis.grid(True, linestyle='--', alpha=0.5)

        # Save the figure.
        for ext in ('svg', 'eps'):
            saved_to = self.target.with_suffix(f'.S_and_n_be.{ext}')
            logging.info(f'Saving plot of n_be to {saved_to}.')
            fig.savefig(saved_to, bbox_inches=0, transparent=False)

        matplot2tikz.save(figure=fig, filepath=self.target.with_suffix(f'.S_and_n_be.tex'))

if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO)

    args = parse_args()

    logging.info(f"Received arguments: {args}")

    runner = StraightLineBenchmark(target = args.target)

    runner.run(args = args.target_args)

    runner.analyse()
