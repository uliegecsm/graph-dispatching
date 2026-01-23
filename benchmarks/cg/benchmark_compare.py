import enum
import functools
import itertools
import json
import logging
import pathlib
import re
import subprocess
import typing

import matplotlib.pyplot
import matplotlib.patches
import numpy
import typeguard

from benchmarks.base import BenchmarkBase, parse_args

class CGFlavor(enum.StrEnum):
    """
    CG solver flavor.
    """
    GRAPH           = 'graph'
    GRAPH_WITH_HOST = 'graph_with_host'
    SINGLEQUEUE     = 'single_queue'

    @property
    @typeguard.typechecked
    def is_graph_based(self) -> bool:
        return self != CGFlavor.SINGLEQUEUE

class CollectionMethod(enum.StrEnum):
    """
    Method used for results collection.
    """
    SINGLE_ITER = 'CollectionMethod::SINGLE_ITER'
    CONVERGENCE = 'CollectionMethod::CONVERGENCE'

    def get(self, x : str):
        match self:
            case CollectionMethod.SINGLE_ITER: return f'{x}single_iter'
            case CollectionMethod.CONVERGENCE: return f'{x}convergence'
            case _:
                raise ValueError()

class CGBenchmark(BenchmarkBase):

    @typeguard.typechecked
    def __init__(self, target: pathlib.Path) -> None:
        super().__init__(target=target)
        self.results = {
            CollectionMethod.CONVERGENCE: pathlib.Path(str(self.target) + '.CONVERGENCE.json'),
            CollectionMethod.SINGLE_ITER: pathlib.Path(str(self.target) + '.SINGLE_ITER.json'),
        }

    @functools.cached_property
    @typeguard.typechecked
    def pairs(self) -> typing.Dict[typing.Tuple[str, str], typing.Tuple[CollectionMethod, CGFlavor]]:
        pairs = {}
        for collection in CollectionMethod:
            for flavor in CGFlavor:
                pairs[((collection, collection.get(str(flavor))))] = (collection, flavor)
        return pairs

    @typeguard.typechecked
    def params(self, *, name : str) -> typing.Tuple[CollectionMethod, CGFlavor, int, int]:
        """
        Retrieve benchmark parameters from its name.
        """
        pattern = rf'CGBenchmark<({"|".join([x[0] for x in self.pairs.keys()])})>/({"|".join([x[1] for x in self.pairs.keys()])})/nrows:([0-9]+)/niters:([0-9]+)/manual_time'

        self.assertRegex(name, pattern)

        match = re.match(pattern, name)

        pair = self.pairs[match.group(1), match.group(2)]

        return *pair, int(match.group(3)), int(match.group(4))

    @typeguard.typechecked
    def run(self, *, args : typing.List[str] = []) -> None:
        """
        Run the benchmark.
        """
        # First, run the benchmark for collecting the timings related to sub-regions, i.e. 'SINGLE_ITER'.
        cmd = [
            self.target,
            '--benchmark_out=' + str(self.results[CollectionMethod.SINGLE_ITER]),
            '--benchmark_out_format=json',
            '--benchmark_min_time=1x',
            '--benchmark_filter=SINGLE_ITER',
            '--benchmark_enable_random_interleaving=true',
            *args,
        ]

        logging.info(f'Running benchmark with {cmd}.')

        subprocess.check_call(cmd)

        # Then, run the benchmark for collecting the timings related to 'CONVERGENCE'.
        cmd = [
            self.target,
            '--benchmark_out=' + str(self.results[CollectionMethod.CONVERGENCE]),
            '--benchmark_out_format=json',
            '--benchmark_min_time=1x',
            '--benchmark_filter=CONVERGENCE',
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
        benchmark_results = {}
        for collection, filename in self.results.items():
            with open(filename, 'r') as fin:
                benchmark_results[collection] = json.load(fin)

        # Collect data for each case.
        data      = {x : {} for x in CollectionMethod}
        time_unit = None

        for collection in CollectionMethod:
            for bench_case in benchmark_results[collection]['benchmarks']:
                logging.info(f'Analysing results of the benchmark case {bench_case['name']} (it ran {bench_case["iterations"]} times, average real time {bench_case["real_time"]} {bench_case["time_unit"]}).')

                if time_unit is None:
                    time_unit = bench_case['time_unit']
                else:
                    self.assertEqual(time_unit, bench_case['time_unit'])

                parsed_collection, flavor, nrows, niters = self.params(name = bench_case['name'])

                self.assertEqual(collection, parsed_collection)

                data[collection][(flavor, nrows)] = {}
                data[collection][(flavor, nrows)]['niters'    ] = niters
                data[collection][(flavor, nrows)]['real_time' ] = bench_case['real_time']

                match collection:
                    case CollectionMethod.SINGLE_ITER:
                        data[collection][(flavor, nrows)]['setup'            ] = bench_case['setup']
                        data[collection][(flavor, nrows)]['create_graph'     ] = bench_case['create-graph']
                        data[collection][(flavor, nrows)]['instantiate_graph'] = bench_case['instantiate-graph']

                        if flavor == CGFlavor.SINGLEQUEUE:
                            raise RuntimeError(f'The {flavor} is not supposed to report for {collection}.')

                        if any(data[collection][(flavor, nrows)][x] == 0 for x in ['setup', 'create_graph', 'instantiate_graph']):
                            raise RuntimeError(f'The {flavor} reported zero for at least one sub-region ({data[collection][(flavor, nrows)]}).')

                    case CollectionMethod.CONVERGENCE:
                        data[collection][(flavor, nrows)]['loop'] = bench_case['loop']
                        self.assertGreater(bench_case['loop'], 0.)

                    case _:
                        raise ValueError(collection)

        # Number of rows set.
        nrows_sorted_set = sorted(set([x[1] for x in data[CollectionMethod.CONVERGENCE].keys()]))
        self.assertEqual(  sorted(set([x[1] for x in data[CollectionMethod.SINGLE_ITER].keys()])), nrows_sorted_set)

        nrows_sorted_set = numpy.asarray(nrows_sorted_set)

        # The baseline is the 'single_queue' flavor.
        BASELINE = CGFlavor.SINGLEQUEUE

        # Collection data for 'CONVERGENCE' (all flavors).
        ratios_full     = {x : [] for x in CGFlavor}
        ratios_per_iter = {x : [] for x in CGFlavor}
        times_per_iter  = {x : [] for x in CGFlavor}

        # Collected data for 'SINGLE_ITER' (graph flavors only).
        times_setup    = {x : [] for x in CGFlavor if x.is_graph_based}
        times_create_g = {x : [] for x in CGFlavor if x.is_graph_based}
        times_instan_g = {x : [] for x in CGFlavor if x.is_graph_based}

        for collection, nrows in itertools.product(CollectionMethod, nrows_sorted_set):

            # All flavors should converge with the same number of iterations.
            if collection == CollectionMethod.CONVERGENCE:
                for flavor in CGFlavor:
                    self.assertEqual(data[collection][(flavor, nrows)]['niters'], data[collection][(BASELINE, nrows)]['niters'])

            for flavor in CGFlavor:
                if collection == CollectionMethod.SINGLE_ITER and not flavor.is_graph_based:
                    continue

                match collection:
                    case CollectionMethod.SINGLE_ITER:
                        flavor_setup    = data[collection][(flavor, nrows)]['setup']
                        flavor_create_g = data[collection][(flavor, nrows)]['create_graph']
                        flavor_instan_g = data[collection][(flavor, nrows)]['instantiate_graph']

                        times_setup   [flavor].append(flavor_setup)
                        times_create_g[flavor].append(flavor_create_g)
                        times_instan_g[flavor].append(flavor_instan_g)

                    case CollectionMethod.CONVERGENCE:
                        baseline_retime = data[collection][(BASELINE, nrows)]['real_time']
                        baseline_loop   = data[collection][(BASELINE, nrows)]['loop']
                        baseline_niters = data[collection][(BASELINE, nrows)]['niters']

                        flavor_retime   = data[collection][(flavor, nrows)]['real_time']
                        flavor_loop     = data[collection][(flavor, nrows)]['loop']

                        ratios_full    [flavor].append(flavor_retime / baseline_retime)
                        ratios_per_iter[flavor].append(flavor_loop   / baseline_loop)
                        times_per_iter [flavor].append(flavor_loop   / baseline_niters)
                    case _:
                        raise ValueError()

        # Consistency check.
        self.assertTrue(all(x == 1. for x in ratios_full    [BASELINE]))
        self.assertTrue(all(x == 1. for x in ratios_per_iter[BASELINE]))

        # Colors for each flavor.
        COLORS = {
            CGFlavor.GRAPH          : 'red',
            CGFlavor.GRAPH_WITH_HOST: 'green',
            CGFlavor.SINGLEQUEUE    : 'blue',
        }

        # Line style for each flavor (for black and white).
        LINESTYLES = {
            CGFlavor.GRAPH          : '-.',
            CGFlavor.GRAPH_WITH_HOST: '--',
            CGFlavor.SINGLEQUEUE    : ':',
        }

        # Markers.
        MARKER_RATIO_FULL     = '^'
        MARKER_RATIO_PER_ITER = 's'
        MARKER_TIMES_PER_ITER = 'o'

        # Create figure for 'CONVERGENCE' related data.
        _, convergence_axes = matplotlib.pyplot.subplots(nrows = 2, ncols = 1, figsize = (10, 7))
        convergence_ax_per_i = convergence_axes[0]
        convergence_ax_ratio = convergence_axes[1]

        convergence_ax_ratio.set_xscale('log')
        convergence_ax_ratio.set_xlabel('size [-]')
        convergence_ax_ratio.set_ylabel('ratio [-]')

        convergence_ax_per_i.set_xscale('log')
        convergence_ax_per_i.set_ylabel(f'avg. iteration [{time_unit}]')

        for flavor in CGFlavor:
            convergence_ax_per_i.plot(
                nrows_sorted_set,
                times_per_iter[flavor],
                marker    = MARKER_TIMES_PER_ITER,
                linestyle = LINESTYLES[flavor] ,
                color     = COLORS[flavor],
                label     = f'{flavor}',
            )

            if flavor == BASELINE or flavor == CGFlavor.GRAPH_WITH_HOST:
                continue

            convergence_ax_ratio.plot(
                nrows_sorted_set,
                ratios_full[flavor],
                marker    = MARKER_RATIO_FULL,
                linestyle = LINESTYLES[flavor],
                color     = COLORS[flavor],
                label     = 'overall',
            )
            convergence_ax_ratio.plot(
                nrows_sorted_set,
                ratios_per_iter[flavor],
                marker    = MARKER_RATIO_PER_ITER,
                linestyle = LINESTYLES[flavor],
                color     = COLORS[flavor],
                label     = 'per iteration',
            )

        # Plot the 'one' ratio.
        convergence_ax_ratio.axhline(y = 1., color = 'black', linestyle = '-')

        # Enable grid.
        convergence_ax_ratio.grid(True)
        convergence_ax_per_i.grid(True)

        # Legend.
        convergence_ax_per_i.legend(framealpha=1., loc = 'upper left')
        convergence_ax_ratio.legend(framealpha=1., loc = 'upper right')

        # Save figure.
        for ext in ('png', 'svg', 'eps'):
            output = self.results[CollectionMethod.CONVERGENCE].with_suffix('.' + ext).resolve()
            logging.info(f'Saving figure to {output}.')
            matplotlib.pyplot.savefig(output, bbox_inches = 0, transparent = True)

        # Plot the time it takes for the CG initial setup, graph creation and instantiation.
        # See also https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#grouped-bar-chart-with-labels.
        EDGECOLOR        = 'black'
        HATCH_SETUP      = ''
        HATCH_CREATE     = '///'
        HATCH_INSTAN     = '\\\\\\'

        _, single_iter_ax = matplotlib.pyplot.subplots(figsize = (10, 7))

        offset = 0

        COMMON = {
            'width'     : 0.25,
            'edgecolor' : EDGECOLOR,
        }

        SCALE = 10

        for flavor in CGFlavor:
            if not flavor.is_graph_based:
                continue

            bottom = numpy.zeros(shape = nrows_sorted_set.shape)

            positions = [x + offset / 4. for x,_ in enumerate(nrows_sorted_set)]

            times_setup[flavor] = numpy.asarray(times_setup[flavor])

            single_iter_ax.bar(positions, times_setup   [flavor] / SCALE, facecolor = COLORS[flavor], hatch = HATCH_SETUP,  bottom = bottom, **COMMON) ; bottom += times_setup   [flavor] / SCALE
            single_iter_ax.bar(positions, times_create_g[flavor],         facecolor = COLORS[flavor], hatch = HATCH_CREATE, bottom = bottom, **COMMON) ; bottom += times_create_g[flavor]
            single_iter_ax.bar(positions, times_instan_g[flavor],         facecolor = COLORS[flavor], hatch = HATCH_INSTAN, bottom = bottom, **COMMON) ; bottom += times_instan_g[flavor]

            offset += 1

        single_iter_ax.set_xticks([x for x, _ in enumerate(nrows_sorted_set)])
        single_iter_ax.set_xticklabels(nrows_sorted_set, rotation = 65)

        single_iter_ax.set_xlabel('size [-]')
        single_iter_ax.set_ylabel(f'time [{time_unit}]')

        # Legend.
        legend = {
            # Flavors.
            'graph'            : matplotlib.patches.Patch(edgecolor = EDGECOLOR, facecolor = COLORS[CGFlavor.GRAPH],           linestyle = ''),
            'graph with host'  : matplotlib.patches.Patch(edgecolor = EDGECOLOR, facecolor = COLORS[CGFlavor.GRAPH_WITH_HOST], linestyle = ''),
            # Type of data.
            f'setup / {SCALE}' : matplotlib.patches.Patch(edgecolor = EDGECOLOR, hatch = HATCH_SETUP,  facecolor = 'white', linestyle = ''),
            'create'           : matplotlib.patches.Patch(edgecolor = EDGECOLOR, hatch = HATCH_CREATE, facecolor = 'white', linestyle = ''),
            'instantiate'      : matplotlib.patches.Patch(edgecolor = EDGECOLOR, hatch = HATCH_INSTAN, facecolor = 'white', linestyle = ''),
        }
        single_iter_ax.legend(
            legend.values(),
            legend.keys(),
            loc = 'upper left',
        )

        # Save figure.
        output = self.results[CollectionMethod.SINGLE_ITER].with_suffix('.svg')
        logging.info(f'Saving figure to {output}.')
        matplotlib.pyplot.savefig(output, bbox_inches = 0, transparent = True)

        output = self.results[CollectionMethod.SINGLE_ITER].with_suffix('.png')
        logging.info(f'Saving figure to {output}.')
        matplotlib.pyplot.savefig(output, bbox_inches = 0, transparent = False)

if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO)

    args = parse_args()

    logging.info(f"Received arguments: {args}")

    runner = CGBenchmark(target = args.target)

    runner.run(args = args.target_args)

    runner.analyse()
