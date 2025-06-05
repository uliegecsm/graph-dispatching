#ifndef GRAPH_DISPATCHING_ALGORITHMS_NEWTON_SOLVER_HPP
#define GRAPH_DISPATCHING_ALGORITHMS_NEWTON_SOLVER_HPP

#include "plog/Log.h"

namespace algorithms::newton {
/**
 * @brief Full step Newton solver.
 *
 * See also https://en.wikipedia.org/wiki/Newton%27s_method#Multidimensional_formulations.
 */
template <typename ProblemType, typename LinearSolverType, typename SubtractType>
struct Solver {
    struct Parameters {
        //! Tolerance on the residual for convergence.
        typename LinearSolverType::mag_t tolerance;

        //! Maximum iterations to perform.
        size_t max_iters;
    };

    ProblemType problem;
    LinearSolverType linear_solver;

    template <Kokkos::ExecutionSpace Exec>
    auto solve(
        const Exec& exec,
        const Parameters& params,
        const typename LinearSolverType::Parameters& params_linear_solver) const {
        //! @todo Promote the initial guess as an argument. For now, it's a vector of zeroes.
        const typename ProblemType::scalar_1d_view_t delta(Kokkos::view_alloc(exec, "delta"), problem.local_sol.size());

        auto res_nrm2 = Kokkos::Experimental::finite_max_v<typename LinearSolverType::mag_t>;
        size_t iter = 0;

        problem.assemble(exec);
        res_nrm2 = KokkosBlas::nrm2(exec, problem.local_rhs);

        while (res_nrm2 > params.tolerance && iter < params.max_iters) {
            PLOG_INFO << "Newton(solve): iteration " << iter;

            [[maybe_unused]] const auto [linear_solver_res_nrm2, linear_solver_num_iters] =
                linear_solver.apply(exec, delta, params_linear_solver);

            Kokkos::parallel_for(
                Kokkos::RangePolicy(exec, 0, problem.params.num_dofs),
                SubtractType{.dst = problem.local_sol, .src = delta});

            problem.assemble(exec);
            res_nrm2 = KokkosBlas::nrm2(exec, problem.local_rhs);

            PLOG_INFO << "Newton(solve): res nrm2 " << res_nrm2;
            ++iter;
        }

        return std::tuple{res_nrm2, iter};
    }
};

} // namespace algorithms::newton

#endif // GRAPH_DISPATCHING_ALGORITHMS_NEWTON_SOLVER_HPP
