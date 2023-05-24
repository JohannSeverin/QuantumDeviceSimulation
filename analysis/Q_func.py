import sys

sys.path.append("../")
from simulation.experiment import SimulationResults

from scipy.special import factorial
import numpy as np
from qutip import qfunc
import matplotlib.pyplot as plt


def Q_of_rho(rhos, x, y):
    n_cutoff = rhos.shape[-1]

    alphas = x + 1j * y

    normalization = np.exp(-alphas * alphas.conj() / 2)

    # Basis change
    ns = np.expand_dims(np.arange(n_cutoff), 1)
    P = normalization * np.power(alphas, ns) / np.sqrt(factorial(ns))

    Q_output = 1 / np.pi * np.einsum("ij, sjk, ki -> si", P.T.conj(), rhos, P)
    return Q_output.real


def qfunc_plotter(results: SimulationResults, interval=10, resolution=100):
    if results.store_states == False:
        raise NotImplementedError("QFuncPlotter only implemented when storing states")

    # TODO - implement sweeps as interactive sliders
    if results.number_of_sweeps > 0:
        raise NotImplementedError("QFuncPlotter only implemented for no sweeps")

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
            Qs[i], extent=[-interval, interval, -interval, interval], cmap="jet"
        )
        axes[i].set_title(f"Q function for state {i}")

    return fig, axes


from ipywidgets import interact, interactive, fixed, interact_manual


def qfunc_with_sweeps_and_time(results: SimulationResults, interval=10, resolution=100):
    time_variable = results.times if not results.only_store_final_state else None

    for i in range(results.number_of_sweeps):
        if results.number_of_sweeps > 1:
            print(f"Plotting sweep {i+1}/{results.number_of_sweeps}")
        qfunc_plotter(results.states[i], interval, resolution)

    return None
