import matplotlib.pyplot as plt
import numpy as np
import os, sys

plt.style.use("../analysis/standard_plot_style.mplstyle")

sys.path.append("../")
from simulation.experiment import SimulationResults


def sweep_analysis(results: SimulationResults):
    if results.number_of_states == 1:
        results.exp_vals = np.expand_dims(results.exp_vals, axis=-1)

    if results.number_of_expvals == 1:
        results.exp_vals = np.expand_dims(results.exp_vals, axis=-1)

    if results.number_of_expvals == 0:
        raise NotImplementedError(
            "Sweep Analysis only implemented for expectation values."
        )

    if results.number_of_sweeps == 1:
        return plot_one_dimensional_sweep(results)

    elif results.number_of_sweeps == 2:
        return plot_two_dimensional_sweep(results)


def plot_one_dimensional_sweep(results: SimulationResults):
    expectation_values = results.exp_vals
    sweep_values = results.sweep_lists
    sweep_param_names = np.array(results.sweep_params).flatten()

    fig, axes = plt.subplots(
        ncols=results.number_of_states,
        nrows=results.number_of_expvals,
        sharex=True,
        sharey=True,
    )

    axes = np.array(axes).reshape(results.number_of_expvals, results.number_of_states)

    # Simple plot
    for i in range(results.number_of_expvals):
        for j in range(results.number_of_states):
            axes[i, j].plot(sweep_values, expectation_values[:, j, i])

    # Fit labels and titles
    for i in range(results.number_of_expvals):
        axes[i, 0].set(ylabel="exp_val")

    for j in range(results.number_of_states):
        axes[0, j].set(title=f"State {j}")
        axes[-1, j].set(xlabel=sweep_param_names[0])

    fig.tight_layout()

    return fig, axes


def plot_two_dimensional_sweep(results: SimulationResults):
    expectation_values = results.exp_vals
    sweep_values = results.sweep_lists
    sweep_param_names = np.array(results.sweep_params).flatten()

    # Expects expectation values with shape (outer_sweep_param_len, inner_sweep_param_len, num_of_states, num_expectation_values)
    fig, axes = plt.subplots(
        ncols=results.number_of_states,
        nrows=results.number_of_expvals,
        sharex=True,
        sharey=True,
    )

    axes = np.array(axes).reshape(results.number_of_expvals, results.number_of_states)

    extent = [
        sweep_values[0][0],
        sweep_values[0][-1],
        sweep_values[1][0],
        sweep_values[1][-1],
    ]

    # Imshow plot
    for i in range(results.number_of_expvals):
        for j in range(results.number_of_states):
            im = axes[i, j].imshow(
                expectation_values[:, :, j, i].T,
                extent=extent,
                origin="lower",
                aspect="auto",
            )
            fig.colorbar(im, ax=axes[i, j])

    # Labels and titles
    for i in range(results.number_of_expvals):
        axes[i, 0].set(ylabel=sweep_param_names[1])

    for j in range(results.number_of_states):
        axes[0, j].set(title=f"State {j}")
        axes[-1, j].set(xlabel=sweep_param_names[0])

    fig.tight_layout()

    return fig, axes


# class Analysis:
#     """
#     Parent class for handling analysis of data from experiments.
#     """

#     def __init__(self, results):

#         self.results = results

# class SweepAnalysis(Analysis):

#     def __init__(self, results):
#         """
#         Sweep Analysis Class to create imshows for 1 or 2D sweeps.
#         """
#         if results.number_of_expvals == 0:
#             raise NotImplementedError("Sweep Analysis only implemented for expectation values.")

#         if results.number_of_sweeps in [1, 2]:
#             self.number_of_sweeps = results.number_of_sweeps # Will be 1 or 2 at the moment

#         elif results.number_of_sweeps == 0:
#             print(f"No sweep made, nothing to plot")

#         else:
#             raise NotImplementedError("Only 1 or 2 sweeps can be plotted")

#         super().__init__(results)

#     # Plot function -> sends to 1D or 2D sweep plots
#     def plot(self):
#         exp_vals     = self.results.exp_vals
#         sweep_list   = self.results.sweep_lists
#         sweep_params = np.array(self.results.sweep_params).flatten()

#         # Increase size if dimensions are 1
#         if self.results.number_of_states == 1:
#             exp_vals = np.expand_dims(exp_vals, axis = -1)

#         if self.results.number_of_expvals == 1:
#             exp_vals = np.expand_dims(exp_vals, axis = -1)

#         # Plotting depending on sweeps
#         if self.results.number_of_sweeps == 1:
#             return self.plot_one_dimensional_sweep(exp_vals, sweep_list, sweep_params)

#         elif self.results.number_of_sweeps == 2:
#             return self.plot_two_dimensional_sweep(exp_vals, sweep_list, sweep_params)

#         else:
#             return None

#     def plot_one_dimensional_sweep(self, expectation_values, sweep_values, sweep_param_names):
#         # Expects expectation values with shape (sweep_param_len, num_of_states, num_expectation_values)
#         fig, axes = plt.subplots(ncols = self.results.number_of_states, nrows = self.results.number_of_expvals, sharex = True, sharey = True)

#         axes = np.array(axes).reshape(self.results.number_of_expvals, self.results.number_of_states)

#         # Simple plot
#         for i in range(self.results.number_of_expvals):
#             for j in range(self.results.number_of_states):
#                 axes[i, j].plot(sweep_values, expectation_values[:, j, i])

#         # Fit labels and titles
#         for i in range(self.results.number_of_expvals):
#             axes[i, 0].set(ylabel = "exp_val")

#         for j in range(self.results.number_of_states):
#             axes[0, j].set(title = f"State {j}")
#             axes[-1, j].set(xlabel = sweep_param_names[0])

#         fig.tight_layout()

#         return fig, axes

#     def plot_two_dimensional_sweep(self, expectation_values, sweep_values, sweep_param_names):
#         # Expects expectation values with shape (outer_sweep_param_len, inner_sweep_param_len, num_of_states, num_expectation_values)
#         fig, axes = plt.subplots(ncols = self.results.number_of_states, nrows = self.results.number_of_expvals, sharex = True, sharey = True)

#         axes = np.array(axes).reshape(self.results.number_of_expvals, self.results.number_of_states)

#         extent = [sweep_values[0][0], sweep_values[0][-1], sweep_values[1][0], sweep_values[1][-1]]

#         # Imshow plot
#         for i in range(self.results.number_of_expvals):
#             for j in range(self.results.number_of_states):
#                 im = axes[i, j].imshow(expectation_values[:, :, j, i].T, extent = extent, origin = "lower", aspect = "auto")
#                 fig.colorbar(im, ax = axes[i, j])

#         # Labels and titles
#         for i in range(self.results.number_of_expvals):
#             axes[i, 0].set(ylabel = sweep_param_names[1])

#         for j in range(self.results.number_of_states):
#             axes[0, j].set(title = f"State {j}")
#             axes[-1, j].set(xlabel = sweep_param_names[0])

#         fig.tight_layout()

#         return fig, axes
