import matplotlib.pyplot as plt
import numpy as np
import os, sys

this_dir = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(this_dir, "standard_plot_style.mplstyle"))

from simulation.experiment import SimulationResults


def sweep_analysis(results: SimulationResults, **kwargs):
    if results.number_of_expvals == 0:
        raise NotImplementedError(
            "Sweep Analysis only implemented for expectation values."
        )

    if results.number_of_sweeps == 1:
        return plot_one_dimensional_sweep(results, **kwargs)

    elif results.number_of_sweeps == 2:
        return plot_two_dimensional_sweep(results, **kwargs)


def plot_one_dimensional_sweep(results: SimulationResults, **kwargs):
    expectation_values = results.exp_vals

    if not results.only_store_final:
        expectation_values = expectation_values[..., -1]

    expectation_values = expectation_values.reshape(
        -1, results.number_of_states, results.number_of_expvals
    )

    sweep_param = results.sweep_parameters[0]
    sweep_values = results.sweep_dict[sweep_param[0]][sweep_param[1]]
    sweep_param_name = results.exp_val_descriptions[0]

    fig, axes = plt.subplots(
        nrows=results.number_of_expvals,
        ncols=results.number_of_states,
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
        axes[-1, j].set(xlabel=sweep_param_name)

    fig.tight_layout()

    return fig, axes


def plot_two_dimensional_sweep(results: SimulationResults, **kwargs):
    expectation_values = results.exp_vals

    if not results.only_store_final:
        expectation_values = expectation_values[..., -1]

    sweep_values = [
        results.sweep_dict[results.sweep_parameters[i][0]][
            results.sweep_parameters[i][1]
        ]
        for i in range(results.number_of_sweeps)
    ]
    sweep_param_names = results.exp_val_descriptions[:2]

    expectation_values = expectation_values.reshape(
        sweep_values[0].shape[0],
        sweep_values[1].shape[0],
        results.number_of_states,
        results.number_of_expvals,
    )

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


def plot_time_evolution(results: SimulationResults):
    expectation_values = results.exp_vals
    times = results.times

    expectation_values = expectation_values.reshape(
        results.number_of_states, results.number_of_expvals, -1
    )

    fig, axes = plt.subplots(
        nrows=results.number_of_expvals,
        ncols=results.number_of_states,
        sharex=True,
        sharey=True,
    )

    axes = np.array(axes).reshape(results.number_of_expvals, results.number_of_states)

    # Simple plot
    for i in range(results.number_of_states):
        for j in range(results.number_of_expvals):
            axes[j, i].plot(times, expectation_values[i, j, :])

    # Fit labels and titles
    for i in range(results.number_of_expvals):
        axes[i, 0].set(ylabel="exp_val")

    for j in range(results.number_of_states):
        axes[0, j].set(title=f"State {j}")
        axes[-1, j].set(xlabel="time")

    fig.tight_layout()

    return fig, axes


def plot_time_evolution_with_single_sweep(results: SimulationResults):
    expectation_values = results.exp_vals

    sweep_param = results.sweep_parameters[0]
    sweep_values = results.sweep_dict[sweep_param[0]][sweep_param[1]]
    sweep_param_name = results.exp_val_descriptions[0]
    times = results.times

    expectation_values = expectation_values.reshape(
        len(sweep_values),
        results.number_of_states,
        results.number_of_expvals,
        len(times),
    )

    fig, axes = plt.subplots(
        ncols=results.number_of_states,
        nrows=results.number_of_expvals,
        sharex=True,
        sharey=True,
    )

    axes = np.array(axes).reshape(results.number_of_expvals, results.number_of_states)

    extent = [
        times[0],
        times[-1],
        sweep_values[0],
        sweep_values[-1],
    ]
    # Imshow plot
    for i in range(results.number_of_expvals):
        for j in range(results.number_of_states):
            # print(expectation_values[:, j, i, :])
            im = axes[i, j].imshow(
                expectation_values[:, j, i, :],
                extent=extent,
                origin="lower",
                aspect="auto",
                interpolation="none",
            )
            fig.colorbar(im, ax=axes[i, j])

    # Labels and titles
    for i in range(results.number_of_expvals):
        axes[i, 0].set(ylabel=sweep_param_name)

    for j in range(results.number_of_states):
        axes[0, j].set(title=f"State {j}")
        axes[-1, j].set(xlabel="time")

    fig.tight_layout()

    return fig, axes
