import enum
import itertools
import json
import logging
import math
import pathlib
import re
import subprocess
import typing

import matplotlib.pyplot
import numpy
import typeguard

from benchmarks.base import BenchmarkBase, parse_args

class Method(enum.StrEnum):
    GRAPH = 'Graph'
    SINGLE_QUEUE = 'SingleQueue'

    @typeguard.typechecked
    def to_strbool(self) -> str:
        match self:
            case Method.GRAPH: return 'true'
            case _:            return 'false'

    @classmethod
    @typeguard.typechecked
    def from_strbool(cls, value : str) -> 'Method':
        match value:
            case 'true' : return Method.GRAPH
            case 'false': return Method.SINGLE_QUEUE
            case _:
                raise ValueError(f'unsupported value {value}')

class Case(typing.NamedTuple):
    method: Method
    detailed_collection: bool
    name: str

class ExtractedConvergence(typing.NamedTuple):
    pcg: list[numpy.ndarray]
    newton: numpy.ndarray

class NewtonBenchmark(BenchmarkBase):

    @typeguard.typechecked
    def __init__(self, target: pathlib.Path) -> None:
        super().__init__(target=target)
        self.results = {
            'details'  : pathlib.Path(str(self.target) + '.details.json'),
            'repeated' : pathlib.Path(str(self.target) + '.repeated.json'),
        }

    @typeguard.typechecked
    def params(self, *, name : str) -> tuple[Case, str]:
        """
        Retrieve benchmark parameters from its name.
        """
        pattern = rf'NewtonBenchmark<(true|false),(true|false)>/([a-z_]+)/(.*)manual_time'

        self.assertRegex(name, pattern)

        match = re.match(pattern, name)

        self.assertEqual(match.group(0), name)

        return Case(
            method=Method.from_strbool(value = match.group(1)),
            detailed_collection=True if match.group(2) == 'true' else False,
            name=match.group(3),
        ), match.group(4)

    @typeguard.typechecked
    def run(self, *, args : typing.List[str]) -> None:
        """
        Run the benchmark.
        """
        # First, run the benchmark for collecting the details of the convergence.
        cmd = [
            self.target,
            '--benchmark_out=' + str(self.results['details']),
            '--benchmark_out_format=json',
            '--benchmark_min_time=1x',
            '--benchmark_enable_random_interleaving=true',
            '--benchmark_filter=.*collect.*',
            *args,
        ]

        logging.info(f'Collecting details of the convergence with {cmd}.')

        subprocess.check_call(cmd)

        # Then, let's compare.
        cmd = [
            self.target,
            '--benchmark_out=' + str(self.results['repeated']),
            '--benchmark_out_format=json',
            '--benchmark_min_time=10s',
            '--benchmark_enable_random_interleaving=true',
            '--benchmark_filter=.*repeat.*',
            *args,
        ]

        logging.info(f'Collecting details of the convergence with {cmd}.')

        subprocess.check_call(cmd)

    @typeguard.typechecked
    def analyse(self) -> None:
        """
        Analyse the results.
        """
        LINESTYLES = {Method.GRAPH : '--', Method.SINGLE_QUEUE : '-.'}

        # Load benchmark results for the details of the convergence, check it made a single iteration.
        logging.info(f'Loading results from {self.results["details"]}.')
        with open(self.results['details'], 'r') as fin:
            self.benchmark_results = json.load(fin)

        fig, ax = matplotlib.pyplot.subplots(nrows = 1, ncols = 1, figsize = (10, 7))

        details = {}

        for bench_case in self.benchmark_results['benchmarks']:
            params, iterations = self.params(name = bench_case['name'])

            logging.info(f'Checking that {bench_case["name"]} ({params}) made a single iteration.')

            self.assertTrue(params.detailed_collection)
            self.assertEqual(iterations, 'iterations:1/')

            self.assertNotIn(params.method, details)

            details[params.method] = {}

            # Parse output files.
            prefix = params.method.to_strbool()
            prefix = self.target.parent / f'NewtonBenchmark_{prefix}_true_/{params.name}'

            solution_file = prefix / 'solution.bin'
            self.assertTrue(solution_file.is_file(), msg = solution_file)

            solution = numpy.fromfile(file = solution_file, dtype = float, sep = '')

            logging.info(f'The solution is stored in {solution_file} and has {solution.shape} shape.')

            ax.plot(solution, label = params.method, linestyle = LINESTYLES[params.method])

            details[params.method]['solution'] = solution

            output_file = prefix / 'output.log'
            self.assertTrue(output_file.is_file(), msg = output_file)

            logging.info(f'Logging has been saved to {output_file}.')

            details[params.method]['output_file'] = output_file

        ax.legend()
        ax.set_title('solution')

        solution_plot = self.target.with_suffix('.solution.svg')
        logging.info(f'Saving solution plot in {solution_plot}.')
        fig.savefig(solution_plot, bbox_inches = 0, transparent = False)

        self.assertTrue(numpy.allclose(
            a = details[Method.GRAPH       ]['solution'],
            b = details[Method.SINGLE_QUEUE]['solution'],
        ), msg = "The solutions should be close.")

        convergence_graph        = self.extract_from_output(file=details[Method.GRAPH       ]['output_file'], method=Method.GRAPH)
        convergence_single_queue = self.extract_from_output(file=details[Method.SINGLE_QUEUE]['output_file'], method=Method.SINGLE_QUEUE)

        self.assertEqual(convergence_graph.newton.shape, convergence_single_queue.newton.shape)
        self.assertTrue(numpy.allclose(convergence_graph.newton, convergence_single_queue.newton))

        logging.info(f'Newton converged in {convergence_graph.newton.shape[0]} iterations.')

        self.assertEqual(len(convergence_graph.       pcg), convergence_graph.newton.shape[0])
        self.assertEqual(len(convergence_single_queue.pcg), convergence_graph.newton.shape[0])

        fig, ax = matplotlib.pyplot.subplots(nrows = 1, ncols = 1, figsize = (10, 7))

        COLORS = iter(itertools.cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown']))

        for outer, (pcg_iter_graph, pcg_iter_queue) in enumerate(zip(convergence_graph.pcg, convergence_single_queue.pcg)):
            self.assertEqual(pcg_iter_graph.shape, pcg_iter_queue.shape, msg = outer)

            color = next(COLORS)

            ax.plot(pcg_iter_graph, label = f'graph - {outer}', color = color, linestyle = LINESTYLES[Method.GRAPH])
            ax.plot(pcg_iter_queue, label = f'queue - {outer}', color = color, linestyle = LINESTYLES[Method.SINGLE_QUEUE])

        ax.legend()
        ax.set_xlabel('iteration [-]')
        ax.set_ylabel('res nrm2 [-]')
        ax.set_yscale('log')

        convergence_plot = self.target.with_suffix('.convergence.svg')
        logging.info(f'Saving convergence plot in {convergence_plot}.')
        fig.savefig(convergence_plot, bbox_inches = 0, transparent = False)

        # Now, retrieve the results for the comparison.
        # We just need a ratio.
        logging.info(f'Loading results from {self.results["repeated"]}.')
        with open(self.results['repeated'], 'r') as fin:
            self.benchmark_results = json.load(fin)

        timing = {x: None for x in Method}
        time_unit = None
        for bench_case in self.benchmark_results['benchmarks']:
            params, iterations = self.params(name = bench_case['name'])

            if not time_unit:
                time_unit = bench_case['time_unit']
            else:
                self.assertEqual(time_unit, bench_case['time_unit'])

            avg = bench_case['real_time']

            logging.info(f'Average time of {bench_case["name"]} ({params}): {avg} [{time_unit}].')

            timing[params.method] = avg

        ratio = timing[Method.SINGLE_QUEUE] / timing[Method.GRAPH]
        logging.info(f'Ratio is {ratio}.')

    @typeguard.typechecked
    def extract_from_output(self, file: pathlib.Path, method: Method) -> ExtractedConvergence:
        """
        Parse the output log to retrieve iterations and residuals.
        """
        with open(file, mode = 'r') as output:
            newton_iter = -1
            pcg_iter    = -1
            newton = []
            pcg    = []
            for line in output:
                if 'Newton(solve): iteration' in line:
                    newton_iter += 1
                    newton.append(math.inf)
                elif f'PCG{method}(apply): starting' in line:
                    pcg_iter += 1
                    pcg.append([])
                else:
                    if (m := re.search(rf'PCG{method}\(apply\): iteration ([0-9]+), res nrm2 ([0-9.e\-]+)', line)) is not None:
                        pcg[pcg_iter].append(float(m.group(2)))
                    elif (m := re.search(r'Newton\(solve\): res nrm2 ([0-9.e\-]+)', line)) is not None:
                        newton[newton_iter] = float(m.group(1))
                    else:
                        raise RuntimeError(line)
            self.assertEqual(newton_iter, pcg_iter)
            return ExtractedConvergence(
                newton=numpy.asarray(newton),
                pcg=[numpy.asarray(x) for x in pcg],
            )

if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO)

    args = parse_args()

    logging.info(f"Received arguments: {args}")

    runner = NewtonBenchmark(target = args.target)

    runner.run(args = args.target_args)

    runner.analyse()
