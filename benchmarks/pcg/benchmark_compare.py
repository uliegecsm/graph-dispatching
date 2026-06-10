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
import numpy.typing
import pandas
import typeguard

from benchmarks.base import BenchmarkBase, parse_args
from benchmarks.common import asymptotic_speedup, n_be, step_like_plot, S_n

class PCGFlavor(enum.StrEnum):
    """
    PCG solver flavor.
    """
    GRAPH = 'graph'
    QUEUE = 'queue'

@dataclasses.dataclass(eq = True, frozen = True)
class Parameters:
    nrows : int
    nsweeps : int

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
        pattern = rf'PCGBenchmark/({"|".join(list(PCGFlavor))})/nrows:([0-9]+)/nsweeps:([0-9]+)/manual_time'

        self.assertRegex(name, pattern)

        match = re.match(pattern, name)

        return PCGFlavor(match.group(1)), Parameters(
            nrows = int(match.group(2)),
            nsweeps = int(match.group(3)),
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
            self.benchmark_results = json.load(fin)

        # Regions of the graph implementation.
        FIELDS = ('graph definition', 'graph instantiation', 'graph submit 0', 'graph submit r')

        # Collect data for each case.
        data = {}
        data[PCGFlavor.QUEUE] = {
            'nrows': [],
            'real_time': [],
            'niters': [],
            'iterations': [],
            'nsweeps': [],
            'loop': [],
        }
        data[PCGFlavor.GRAPH] = {
            'nrows': [],
            'real_time': [],
            'niters': [],
            'iterations': [],
            'nsweeps': [],
            'graph definition': [],
            'graph instantiation': [],
            'graph submit 0': [],
            'graph submit r': [],
        }

        time_unit = None

        for bench_case in self.benchmark_results['benchmarks']:
            logging.info(f'Analysing results of the benchmark case {bench_case["name"]} (it ran {bench_case["iterations"]} times, average real time {bench_case["real_time"]} {bench_case["time_unit"]}).')

            if time_unit is None:
                time_unit = bench_case['time_unit']
            else:
                self.assertEqual(time_unit, bench_case['time_unit'])

            flavor, params = self.params(name = bench_case['name'])

            data[flavor]['nrows'].append(params.nrows)
            data[flavor]['real_time'].append(bench_case['real_time'])
            data[flavor]['iterations'].append(bench_case['iterations'])
            data[flavor]['niters'].append(bench_case['num_iters'])
            data[flavor]['nsweeps'].append(params.nsweeps)

            match flavor:
                case PCGFlavor.GRAPH:
                    for field in FIELDS:
                        self.assertGreater(bench_case[field], 0)
                        data[flavor][field].append(bench_case[field])

                case PCGFlavor.QUEUE:
                    self.assertGreater(bench_case['loop'], 0.)
                    data[flavor]['loop'].append(bench_case['loop'])

                case _:
                    raise ValueError(flavor)

        # Transform to Pandas dataframes.
        data = {flavor: pandas.DataFrame(data[flavor]) for flavor in PCGFlavor}

        # Number of sweeps (unique, sorted).
        nsweeps_sorted_set = sorted(data[PCGFlavor.GRAPH]['nsweeps'].unique())
        self.assertEqual(len(nsweeps_sorted_set), 3)

        # Number of rows (unique, sorted).
        nrows_sorted_set = sorted(data[PCGFlavor.GRAPH]['nrows'].unique())
        nrows_sorted_set = numpy.asarray(nrows_sorted_set)

        # Sort according to increasing number of sweeps and rows.
        for flavor in PCGFlavor:
            data[flavor] = data[flavor].sort_values(by=['nrows', 'nsweeps']).reset_index(drop=True)

        # Normalize the 'queue' loop time w.r.t. the number of iterations.
        data[PCGFlavor.QUEUE]['T_E'] = data[PCGFlavor.QUEUE]['loop'] / data[PCGFlavor.QUEUE]['niters']

        # Compute T_G_s0, T_G_sr, asymptotic speedup, n_be and S_n.
        data[PCGFlavor.GRAPH]['T_G_s0'] = data[PCGFlavor.GRAPH]['graph submit 0']
        data[PCGFlavor.GRAPH]['T_G_sr'] = data[PCGFlavor.GRAPH]['graph submit r'] / (data[PCGFlavor.GRAPH]['niters'] - 1)
        data[PCGFlavor.GRAPH]['S'] = asymptotic_speedup(T_E=data[PCGFlavor.QUEUE]['T_E'], T_G_sr=data[PCGFlavor.GRAPH]['T_G_sr'])
        data[PCGFlavor.GRAPH]['n_be'] = n_be(
            T_E=data[PCGFlavor.QUEUE]['T_E'],
            T_G_d=data[PCGFlavor.GRAPH]['graph definition'],
            T_G_i=data[PCGFlavor.GRAPH]['graph instantiation'],
            T_G_s0=data[PCGFlavor.GRAPH]['T_G_s0'],
            T_G_sr=data[PCGFlavor.GRAPH]['T_G_sr'],
        )
        data[PCGFlavor.GRAPH]['S_n'] = S_n(
            n_T_E=data[PCGFlavor.QUEUE]['loop'],
            T_G_d=data[PCGFlavor.GRAPH]['graph definition'],
            T_G_i=data[PCGFlavor.GRAPH]['graph instantiation'],
            T_G_s0=data[PCGFlavor.GRAPH]['T_G_s0'],
            n_m1_T_G_sr=data[PCGFlavor.GRAPH]['graph submit r'],
        )
        data[PCGFlavor.GRAPH]['n_be_integer'] = data[PCGFlavor.GRAPH]['n_be'].apply(numpy.ceil).astype(int)

        # Debug logging.
        for flavor in PCGFlavor:
            logging.info(f'Collected data for flavor {flavor}:\n{data[flavor]}')

        # Create the figure.
        fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1)

        data_graph = data[PCGFlavor.GRAPH]

        # Plot the speedup after n submissions (S_n) and the asymptotic speedup (S) (left axis).
        COLOR_S = 'tab:red'
        ax_S = ax

        ax_S_min = numpy.floor(data_graph['S'].min() * 2) / 2
        ax_S_max = numpy.ceil (data_graph['S'].max() * 2) / 2

        STYLES = {('-', 'o'), ('--', '^'), (':', 's')}

        for nsweeps, (linestyle, marker) in zip(nsweeps_sorted_set, STYLES, strict=True):
            data_nrows = data[PCGFlavor.GRAPH][data[PCGFlavor.GRAPH]['nsweeps'] == nsweeps]['nrows'].values
            data_S = data[PCGFlavor.GRAPH][data[PCGFlavor.GRAPH]['nsweeps'] == nsweeps]['S'].values
            ax_S.plot(data_nrows, data_S, linestyle=linestyle, marker=marker, color=COLOR_S)

        ax_S.set_ylabel('$S$ [-]', color=COLOR_S)
        ax.hlines(y=1, xmin=min(nrows_sorted_set), xmax=max(nrows_sorted_set), color='black')
        ax_S.set_xlabel('Number of rows [-]')
        ax_S.tick_params(axis='y', labelcolor=COLOR_S)
        ax_S.set_xlim(min(nrows_sorted_set), max(nrows_sorted_set))
        ax_S.set_ylim(ax_S_min, ax_S_max)
        ax_S.set_xscale('log')
        ax_S.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, _: f'{x:.1f}'.rstrip('0').rstrip('.')
        ))

        # Plot 'n_be' where it makes sense (i.e. where S >= 1) (right axis).
        COLOR_N_BE = 'tab:blue'

        nsweeps_for_plot = 8
        data_nrows = data[PCGFlavor.GRAPH][data[PCGFlavor.GRAPH]['nsweeps'] == nsweeps_for_plot]['nrows'].values
        data_S = data[PCGFlavor.GRAPH][data[PCGFlavor.GRAPH]['nsweeps'] == nsweeps_for_plot]['S'].values
        data_n_be_integer = data[PCGFlavor.GRAPH][data[PCGFlavor.GRAPH]['nsweeps'] == nsweeps_for_plot]['n_be_integer'].values

        if (mask := data_n_be_integer > 0).any():
            ax_n_be = ax_S.twinx()
            step_like_plot(
                ax=ax_n_be,
                x=data_nrows,
                y=data_n_be_integer,
                predicate=mask,
                color=COLOR_N_BE,
            )
            ax_n_be.set_ylabel(r'$n_{\text{be}}$ [-]', color=COLOR_N_BE)
            ax_n_be.tick_params(axis='y', labelcolor=COLOR_N_BE)

            n_be_integer_max = int(data_n_be_integer[mask].max())
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
            saved_to = self.target.with_suffix(f'.{PCGFlavor.GRAPH}.{nsweeps_for_plot}._and_n_be.{ext}')
            logging.info(f'Saving plot of n_be to {saved_to}.')
            fig.savefig(saved_to, bbox_inches=0, transparent=False)

        matplot2tikz.save(figure=fig, filepath=self.target.with_suffix(f'.{PCGFlavor.GRAPH}.{nsweeps_for_plot}.S_and_n_be.tex'))

if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO)

    args = parse_args()

    logging.info(f"Received arguments: {args}")

    runner = PCGBenchmark(target = args.target)

    runner.run(args = args.target_args)

    runner.analyse()
