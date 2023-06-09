import sys

sys.path.append("../")
from simulation.experiment import SimulationResults

from scipy.special import factorial
import numpy as np
from qutip import qfunc, ket2dm
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

colormaps = [
    LinearSegmentedColormap.from_list("mycmap", ["white", f"C{i}"]) for i in range(10)
]


def Q_of_rho(rhos, x, y, rotate=0):
    n_cutoff = rhos.shape[-1]

    alphas = x + 1j * y

    if rotate != 0:
        alphas *= np.exp(-1j * rotate * np.pi * 2)

    normalization = np.exp(-alphas * alphas.conj() / 2)

    # Basis change
    ns = np.expand_dims(np.arange(n_cutoff), 1)
    P = normalization * np.power(alphas, ns) / np.sqrt(factorial(ns))

    Q_output = 1 / np.pi * np.einsum("ij, sjk, ki -> si", P.T.conj(), rhos, P)
    return Q_output.real


def qfunc_plotter(results: SimulationResults, interval=10, resolution=100):
    states = results.states

    # Increase size if dimensions are 1
    if results.number_of_states == 1:
        states = np.expand_dims(states, axis=-3)

    # DO A PTRACE
    qubit_dims, resonator_dims = (
        results.dimensions["qubit"],
        results.dimensions["resonator"],
    )

    reshaped = np.reshape(
        states, (-1, qubit_dims, resonator_dims, qubit_dims, resonator_dims)
    )
    ptraced = np.einsum("ijklm -> ikm", reshaped)

    # Calculate Q functions
    x = np.linspace(-interval, interval, resolution)
    y = np.linspace(-interval, interval, resolution)

    X, Y = np.meshgrid(x, y)

    Qs = Q_of_rho(ptraced, X.flatten(), Y.flatten()).reshape(-1, resolution, resolution)

    # Setup figure
    fig, axes = plt.subplots(ncols=results.number_of_states, figsize=(10, 10))

    if results.number_of_states == 1:
        axes = np.array([axes])

    # Create the plots

    for i in range(results.number_of_states):
        axes[i].imshow(
            Qs[i], extent=[-interval, interval, -interval, interval], cmap=colormaps[i]
        )
        axes[i].set_title(f"Q function for state {i}")

    return fig, axes


from ipywidgets import interact, interactive, fixed, interact_manual


def qfunc_plotter_with_time_slider(
    results: SimulationResults,
    interval=10,
    resolution=100,
    time_steps=1,
    demod_frequency=0,
):
    if results.only_store_final:
        return qfunc_plotter(results, interval, resolution)

    qubit_dims, resonator_dims = (
        results.dimensions["qubit"],
        results.dimensions["resonator"],
    )

    states = results.states

    if states.shape[-1] == states.shape[-2] == qubit_dims * resonator_dims:
        pass
    else:
        states = np.array(
            [
                np.kron(states[i][t], states[i][t])
                for i in range(results.number_of_states)
                for t in range(len(results.times))
            ]
        )
    # Increase size if dimensions are 1
    if results.number_of_states == 1:
        states = np.expand_dims(states, axis=-4)

    # DO A PTRACE

    reshaped = np.reshape(
        states,
        (
            results.number_of_states,
            len(results.times),
            qubit_dims,
            resonator_dims,
            qubit_dims,
            resonator_dims,
        ),
    )
    ptraced = np.einsum("tijklm -> tikm", reshaped)

    def plotter(time):
        idx = np.argmin(np.abs(results.times - time))
        x = np.linspace(-interval, interval, resolution)
        y = np.linspace(-interval, interval, resolution)

        X, Y = np.meshgrid(x, y)

        Qs = Q_of_rho(
            ptraced[:, idx, ...],
            X.flatten(),
            Y.flatten(),
            rotate=demod_frequency * results.times[idx],
        ).reshape(-1, resolution, resolution)

        # Setup figure
        fig, axes = plt.subplots(ncols=results.number_of_states, figsize=(10, 10))

        if results.number_of_states == 1:
            axes = np.array([axes])

        # Create the plots

        for i in range(results.number_of_states):
            axes[i].imshow(
                Qs[i],
                extent=[-interval, interval, -interval, interval],
                cmap=colormaps[i],
            )
            axes[i].set_title(f"Q function for state {i}")

        return fig, axes

    return interact(plotter, time=(results.times[0], results.times[-1], time_steps))
