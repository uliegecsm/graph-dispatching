import dataclasses
import json
import logging
import pathlib
import re
import subprocess
import typing

import numpy
import numpy.testing
import pandas
import typeguard

from benchmarks.base import BenchmarkBase, parse_args
from benchmarks.graph.common import Method, asymptotic_speedup, n_be


@dataclasses.dataclass(frozen=True)
class Parameters:
    method: Method
    num_branches: int
    num_submits: int
    num_nodes: int

class ForkBenchmark(BenchmarkBase):

    @typeguard.typechecked
    def __init__(self, target: pathlib.Path) -> None:
        super().__init__(target=target)
        self.results = f'{self.target}.json'

    @typeguard.typechecked
    def params(self, *, name: str) -> Parameters:
        """
        Retrieve benchmark parameters from its name.
        """
        pattern = rf'ForkBenchmark/({"|".join(Method)})/num_branches:([0-9]+)/num_submits:([0-9]+)/num_nodes:([0-9]+)'

        self.assertRegex(name, pattern)

        match = re.match(pattern, name)

        return Parameters(
            method=Method(match.group(1)),
            num_branches=int(match.group(2)),
            num_submits=int(match.group(3)),
            num_nodes=int(match.group(4)),
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
            'num_branches': [],
            'num_submits': [],
            'num_nodes': [],
            'real_time': [],
        }
        data[Method.GRAPH] = {
            'num_branches': [],
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

            data[params.method]['num_branches'].append(params.num_branches)
            data[params.method]['num_nodes'].append(params.num_nodes)
            data[params.method]['num_submits'].append(params.num_submits)
            data[params.method]['real_time'].append(bench_case['real_time'])

            if params.method == Method.GRAPH:
                for field in ('create', 'instantiate', 'per_submit', 'all_submits'):
                    data[params.method][field].append(bench_case[field])

        # Transform to Pandas dataframes.
        data_exec = pandas.DataFrame(data[Method.EXEC])
        data_graph = pandas.DataFrame(data[Method.GRAPH])

        # Assert the available node and branch counts match.
        numpy.testing.assert_array_equal(numpy.sort(data_exec['num_nodes'].unique()), numpy.sort(data_graph['num_nodes'].unique()))
        numpy.testing.assert_array_equal(numpy.sort(data_exec['num_branches'].unique()), numpy.sort(data_graph['num_branches'].unique()))

        # Sort according to increasing num of branches, and increasing number of nodes.
        SORT_BY: typing.Final[list[str]] = ['num_branches', 'num_nodes']
        data_exec = data_exec.sort_values(by=SORT_BY).reset_index(drop=True)
        data_graph = data_graph.sort_values(by=SORT_BY).reset_index(drop=True)

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

if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO)

    args = parse_args()

    logging.info(f"Received arguments: {args}")

    runner = ForkBenchmark(target = args.target)

    runner.run(args = args.target_args)

    runner.analyse()
