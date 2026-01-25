#ifndef GRAPH_DISPATCHING_APPS_HEAT_NONLINEAR1DHEATTRANSFER_HPP
#define GRAPH_DISPATCHING_APPS_HEAT_NONLINEAR1DHEATTRANSFER_HPP

#include "kokkos-utils/view/slice.hpp"

namespace apps::heat {

template <typename scalar_3d_view_t, typename scalar_2d_view_t, typename scalar_1d_view_t>
struct Scatter {
    scalar_3d_view_t stacked_elem_matrices;
    scalar_2d_view_t stacked_elem_rhss;

    scalar_1d_view_t local_matrix_values;
    scalar_1d_view_t local_rhs_values;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T elm_id) const {
        using Kokkos::utils::view::slice;

        const auto elm_matrix = slice<3>(stacked_elem_matrices, elm_id);
        const auto elm_rhs = slice<2>(stacked_elem_rhss, elm_id);

        Kokkos::atomic_add(&local_matrix_values(3 * elm_id), elm_matrix(0, 0));
        Kokkos::atomic_add(&local_matrix_values(3 * elm_id + 1), elm_matrix(0, 1));
        Kokkos::atomic_add(&local_matrix_values(3 * elm_id + 2), elm_matrix(1, 0));
        Kokkos::atomic_add(&local_matrix_values(3 * elm_id + 3), elm_matrix(1, 1));

        Kokkos::atomic_add(&local_rhs_values(elm_id), elm_rhs(0));
        Kokkos::atomic_add(&local_rhs_values(elm_id + 1), elm_rhs(1));

        //! @todo To be moved to a proper Dirichlet BC functor.
        if (elm_id == 0) {
            local_matrix_values(0) = 1.;
            local_rhs_values(0) = 0.;
            local_matrix_values(1) = 0.;
            local_matrix_values(2) = 0.;
        }
    }
};

/**
 * Let's consider the following non-linear 1D heat transfer equation in \f$ T(x) : x \in [0, L] \to \mathbb{R}\f$:
 * \f[
 *      \cfrac{d^2 T}{dx^2} - k T^2 = 0
 * \f]
 * with the following boundary conditions:
 * \f[
 *      \begin{cases}
 *          T(0) = 1
 *          \cfrac{dT}{dx}(L) = 0
 *          k \in real numbers
 *      \end{cases}
 * \f]
 *
 * The discretization uses order 1 basis functions.
 */
template <Kokkos::MemorySpace MemorySpace, Kokkos::ExecutionSpace ExecutionSpace, bool UseGraph>
struct NonLinear1DHeatTransfer {
    using scalar_t = double;

    //! The @c KokkosSparse::CrsMatrix requires a signed ordinal type.
    using local_ordinal_t = int;

    using scalar_1d_view_t = Kokkos::View<scalar_t*, MemorySpace>;
    using scalar_2d_view_t = Kokkos::View<scalar_t**, MemorySpace>;
    using scalar_3d_view_t = Kokkos::View<scalar_t***, MemorySpace>;
    using local_matrix_t = KokkosSparse::CrsMatrix<scalar_t, local_ordinal_t, MemorySpace>;
    using local_graph_t = typename local_matrix_t::staticcrsgraph_type;
    using local_graph_row_map_t = typename local_graph_t::row_map_type::non_const_type;
    using local_graph_entries_t = typename local_graph_t::entries_type::non_const_type;
    using local_matrix_values_t = typename local_matrix_t::values_type;

    using scatter_t = Scatter<scalar_3d_view_t, scalar_2d_view_t, scalar_1d_view_t>;

    using graph_t = std::conditional_t<UseGraph, Kokkos::Experimental::Graph<ExecutionSpace>, bool>;

    struct Parameters {
        //! @name Given parameters.
        ///@{
        local_ordinal_t num_elems = 0;
        scalar_t length = 1.;
        scalar_t k = 1.;
        ///@}

        //! @name Deduced parameters.
        ///@{
        local_ordinal_t num_dofs = num_elems + 1;
        scalar_t h = length / num_elems;
        ///@}
    };

    template <typename ViewType>
    struct InitialGuess {
        ViewType guess;
        typename ViewType::value_type step;

        template <Kokkos::ExecutionSpace Exec>
        void apply(const Exec& exec) const {
            Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, guess.size()), *this);
        }

        template <std::integral T>
        KOKKOS_FUNCTION void operator()(const T idx) const {
            guess(idx) = 1. - idx * step;
        }
    };

    template <Kokkos::ExecutionSpace Exec>
    NonLinear1DHeatTransfer(const Exec& exec, Parameters params_)
        : params(std::move(params_)) // NOLINT(performance-move-const-arg)
    {
        this->init(exec);
    }

    template <Kokkos::ExecutionSpace Exec>
    void init(const Exec& exec) {
        //! Build the row map and entries of the graph of the local matrix.
        const auto num_entries = 2 * 2 + (params.num_dofs - 2) * 3;

        local_graph_row_map_t row_map(
            Kokkos::view_alloc(exec, "local graph - row map", Kokkos::WithoutInitializing),
            params.num_dofs + 1); // NOLINT(misc-const-correctness)
        Kokkos::deep_copy(exec, Kokkos::subview(row_map, 0), 0);
        Kokkos::parallel_for(
            Kokkos::RangePolicy(exec, 1, params.num_dofs),
            KOKKOS_LAMBDA(const local_ordinal_t idx) { row_map(idx) = 2 + (idx - 1) * 3; });
        Kokkos::deep_copy(exec, Kokkos::subview(row_map, row_map.size() - 1), num_entries);

        local_graph_entries_t entries(
            Kokkos::view_alloc(exec, "local graph - entries", Kokkos::WithoutInitializing),
            num_entries); // NOLINT(misc-const-correctness)
        Kokkos::deep_copy(exec, Kokkos::subview(entries, 0), 0);
        Kokkos::deep_copy(exec, Kokkos::subview(entries, 1), 1);
        Kokkos::parallel_for(
            Kokkos::RangePolicy(exec, 1, params.num_dofs - 1), KOKKOS_LAMBDA(const local_ordinal_t idx) {
                entries(2 + (idx - 1) * 3) = idx - 1;
                entries(2 + (idx - 1) * 3 + 1) = idx;
                entries(2 + (idx - 1) * 3 + 2) = idx + 1;
            });
        Kokkos::deep_copy(exec, Kokkos::subview(entries, num_entries - 2), params.num_dofs - 2);
        Kokkos::deep_copy(exec, Kokkos::subview(entries, num_entries - 1), params.num_dofs - 1);

        //! Build the local matrix.
        this->local_matrix = local_matrix_t{
            "local matrix",
            params.num_dofs,
            local_matrix_values_t(
                Kokkos::view_alloc(Kokkos::WithoutInitializing, "local matrix - values", exec), num_entries),
            local_graph_t{std::move(entries), std::move(row_map)}  // NOLINT(performance-move-const-arg)
        };

        //! Build the local RHS.
        this->local_rhs =
            scalar_1d_view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "local rhs", exec), params.num_dofs);

        /// Build the local solution placeholder. We'll use it as the initial guess, so we put a 1 for the first dof
        /// to match the Dirichlet BC on the left.
        scalar_1d_view_t local_sol_(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "local sol"), params.num_dofs);
        InitialGuess{.guess = local_sol_, .step = params.length / params.num_elems}.apply(exec);
        this->local_sol = std::move(local_sol_);

        //! Create the stacked views.
        this->stacked_elem_matrices = scalar_3d_view_t(
            Kokkos::view_alloc(exec, "stacked elm matrices", Kokkos::WithoutInitializing), params.num_elems, 2, 2);
        this->stacked_elem_rhss = scalar_2d_view_t(
            Kokkos::view_alloc(exec, "stacked elm rhss", Kokkos::WithoutInitializing), params.num_elems, 2);
    }

    //! Fill Jacobian and residual for element @c ielem.
    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T ielem) const {
        using Kokkos::utils::view::slice;

        const auto elem_mat = slice<3>(stacked_elem_matrices, ielem);
        const auto elem_rhs = slice<2>(stacked_elem_rhss, ielem);

        const auto idof_l = ielem;
        const auto idof_r = ielem + 1;

        //! For readability.
        const auto sol_l = local_sol(idof_l);
        const auto sol_r = local_sol(idof_r);

        //! Residual.
        const auto contr_stiff = (sol_l - sol_r) / params.h;
        elem_rhs(0) = contr_stiff
                    + params.h * params.k
                          * (1. / 4. * sol_l * sol_l + 1. / 6. * sol_l * sol_r + 1. / 12. * sol_r * sol_r);
        elem_rhs(1) = -contr_stiff
                    + params.h * params.k
                          * (1. / 12. * sol_l * sol_l + 1. / 6. * sol_l * sol_r + 1. / 4. * sol_r * sol_r);

        //! Jacobian.
        elem_mat(0, 0) = 1. / params.h + params.h * params.k * (sol_l / 2. + sol_r / 6.);
        elem_mat(1, 1) = 1. / params.h + params.h * params.k * (sol_l / 6. + sol_r / 2.);
        elem_mat(0, 1) = -1. / params.h + params.h * params.k * (sol_l / 6. + sol_r / 6.);

        elem_mat(1, 0) = elem_mat(0, 1);
    }

    template <Kokkos::ExecutionSpace Exec>
    requires(!UseGraph)
    void assemble(const Exec& exec) const {
        Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, params.num_elems), *this);

        Kokkos::deep_copy(exec, local_rhs, 0.);
        Kokkos::deep_copy(exec, local_matrix.values, 0.);

        Kokkos::parallel_for(
            Kokkos::RangePolicy(exec, 0, params.num_elems),
            scatter_t{
                .stacked_elem_matrices = stacked_elem_matrices,
                .stacked_elem_rhss = stacked_elem_rhss,
                .local_matrix_values = local_matrix.values,
                .local_rhs_values = local_rhs});
    }

    template <Kokkos::ExecutionSpace Exec>
    requires(UseGraph)
    void assemble(const Exec& exec) const {
        if (!this->graph) {
            this->graph.emplace(exec);

            auto node_fill = this->graph->root_node()
                                 .then_parallel_for(Kokkos::RangePolicy(exec, 0, params.num_elems), *this);

            auto node_local_rhs_reset = ::algorithms::pcg::then_deep_copy(this->graph->root_node(), local_rhs, 0.);
            auto node_local_mat_reset =
                ::algorithms::pcg::then_deep_copy(this->graph->root_node(), local_matrix.values, 0.);

            Kokkos::Experimental::when_all(
                std::move(node_fill), std::move(node_local_rhs_reset), std::move(node_local_mat_reset))
                .then_parallel_for(
                    Kokkos::RangePolicy(exec, 0, params.num_elems),
                    scatter_t{
                        .stacked_elem_matrices = stacked_elem_matrices,
                        .stacked_elem_rhss = stacked_elem_rhss,
                        .local_matrix_values = local_matrix.values,
                        .local_rhs_values = local_rhs});
        }

        this->graph->submit(exec);
    }

    Parameters params;

    local_matrix_t local_matrix;
    scalar_1d_view_t local_rhs;
    scalar_1d_view_t local_sol;

    scalar_3d_view_t stacked_elem_matrices;
    scalar_2d_view_t stacked_elem_rhss;

   protected:
    //! It will be lazily constructed in @ref assemble, so it must be @c mutable.
    mutable std::optional<graph_t> graph = std::nullopt;
};

} // namespace apps::heat

#endif // GRAPH_DISPATCHING_APPS_HEAT_NONLINEAR1DHEATTRANSFER_HPP
