## Setup
import numpy as np
import matplotlib.pyplot as plt

import os, sys, pickle

sys.path.append("..")

# Paths and imports
experiment_path = "/mnt/c/Users/johan/OneDrive/Skrivebord/QDS_data"

# Load devices
from devices.device import Transmon, Resonator
from devices.system import (
    dispersive_shift,
    QubitResonatorSystem,
)
from devices.pulses import GaussianPulse, SquareCosinePulse


## Define devices
qubit = Transmon(
    EC=15 / 100 * 2 * np.pi,
    EJ=15 * 2 * np.pi,
    n_cutoff=15,
    levels=4,
    ng=0.0,
    T1=0,
)

resonator = Resonator(
    frequency=6.00, levels=10, kappa=1 / 10
)  # 0.00125 * 2 * np.pi * 3)

from devices.device import Device

coupling_strength = 0.25 * 2 * np.pi

qubit_pulse = None

resonator_pulse = SquareCosinePulse(
    amplitude=0.15,
    frequency=6.045,
    phase=-np.pi / 2,
    duration=100,
    start_time=0,
)

omega_minus = 2 * np.pi * (resonator.frequency - resonator_pulse.frequency)

amplitude = (
    -coupling_strength
    * resonator_pulse.amplitude
    / np.sqrt(omega_minus**2 + resonator.kappa**2 / 4)
)
phase = np.arctan(-2 * omega_minus / resonator.kappa)

from devices.pulses import CloakedPulse, CloakedPulse_with_x_pulse

qubit_pulse_w_x = CloakedPulse_with_x_pulse(
    frequency=resonator_pulse.frequency,
    resonator_frequency=resonator.frequency,
    amplitude=np.array([0, amplitude]),
    duration=100,
    start_time=0,
    kappa=resonator.kappa,
    phase=phase,
    x_frequency=(4.08677033 - 0.06209304333366096) * np.linspace(0.98, 1.01, 5),
    x_duration=16,
    x_ampltiude=np.pi / 16 / 1.305,
)


## Define System
system = QubitResonatorSystem(
    qubit=qubit,
    resonator=resonator,
    qubit_pulse=qubit_pulse_w_x,
    resonator_pulse=resonator_pulse,
    coupling_strength=coupling_strength,
)

## Simulate
from simulation.experiment import (
    LindbladExperiment,
    MonteCarloExperiment,
    SchroedingerExperiment,
)

times = np.linspace(0, 100, 500)

experiment = LindbladExperiment(
    system=system,
    states=[system.get_states(0, 0)],
    times=times,
    store_states=True,
    only_store_final=True,
    expectation_operators=[
        system.photon_number_operator(),
        system.qubit_state_occupation_operator(1),
    ],
    # ntraj=1,
    # exp_val_method="average",
)

results = experiment.run()


results.save_path = os.path.join(experiment_path, "cloak_drive")
# results.save()

from analysis.auto import automatic_analysis

from analysis.Q_func import qfunc_plotter, qfunc_plotter_with_time_slider

automatic_analysis(results)


# qfunc_plotter_with_time_slider(
#     results, interval=4.0, demod_frequency=resonator_pulse.frequency
# # )

plt.figure()
times = np.linspace(0, 100, 5000)
pulse_shape_reso = [resonator_pulse.pulse(t, None) for t in times]
pulse_shape_cloak = [qubit_pulse_w_x.pulse(t, None) for t in times]
plt.plot(times, pulse_shape_reso, label="Resonator pulse")

plt.plot(times, pulse_shape_cloak, label="Cloaking pulse")


# plt.figure()
# alpha_ground = results.exp_vals[0, 1, :] + 1j * results.exp_vals[0, 2, :]
# alpha_excited = results.exp_vals[1, 1, :] + 1j * results.exp_vals[1, 2, :]

# convolving = np.exp(1j * 2 * np.pi * resonator_pulse.frequency * results.times)

# demod_ground = alpha_ground * convolving
# demod_excited = alpha_excited * convolving

# I_ground, Q_ground = np.real(demod_ground), np.imag(demod_ground)
# I_excited, Q_excited = np.real(demod_excited), np.imag(demod_excited)

# plt.plot(I_ground, Q_ground, label="Ground")
# plt.plot(I_excited, Q_excited, label="Excited")


# for i in range(3):
#     plt.figure()
#     alpha_ground = results.exp_vals[i, 0, 1, :] + 1j * results.exp_vals[i, 0, 2, :]
#     alpha_excited = results.exp_vals[i, 1, 1, :] + 1j * results.exp_vals[i, 1, 2, :]

#     convolving = np.exp(1j * 2 * np.pi * resonator_pulse.frequency * results.times)

#     demod_ground = alpha_ground * convolving
#     demod_excited = alpha_excited * convolving

#     I_ground, Q_ground = np.real(demod_ground), np.imag(demod_ground)
#     I_excited, Q_excited = np.real(demod_excited), np.imag(demod_excited)

#     plt.plot(I_ground, Q_ground, label="Ground")
#     plt.plot(I_excited, Q_excited, label="Excited")

# fig, ax = plt.subplots(1, 2)
# fig.suptitle("Drive on Resonator Frequency")

# envelope = lambda t: 1.02991037 * np.tanh(
#     qubit_pulse.scale * (t - qubit_pulse.start_time)
# )

# ax[0].plot(results.times, results.exp_vals[0, 0, 0, 0, :], "k--", label="No coupling")
# ax[0].plot(results.times, envelope(results.times), label="Cloak Envelope")

# ax[0].legend(fontsize=12, loc="upper left")

# ax[1].plot(
#     results.times,
#     results.exp_vals[0, 0, 0, 0, :],
#     label="No coupling",
#     ls="--",
#     color="k",
# )

# for i, strength in enumerate(results.sweep_dict["qubit_pulse"]["amplitude"]):
#     relative_strength = strength / coupling_strength
#     ax[1].plot(
#         results.times,
#         results.exp_vals[1, i, 0, 0, :],
#         label=f"Cloak drive - strength: {int(abs(relative_strength)):}",
#     )

# ax[1].legend(fontsize=12, loc="upper left")
