## Setup
import numpy as np

import sys

sys.path.append("..")

experiment_name = "control_test"
experiment_path = "/mnt/c/Users/johan/OneDrive/Skrivebord/QDS_data/" + experiment_name


# Load devices/system
from devices.device import Resonator, Transmon
from devices.pulses import SquareCosinePulse, GaussianPulse
from devices.system import QubitResonatorSystem, dispersive_shift

# Load Simulation Experiment
from simulation.experiment import SchroedingerExperiment

# load Analysis tool
from analysis.auto import automatic_analysis


times = np.linspace(0, 32, 3200)

## Define devices
qubit = Transmon(
    EC=15 / 100 * 2 * np.pi, EJ=15 * 2 * np.pi, n_cutoff=15, levels=4, ng=0.0
)

resonator = Resonator(6.02, levels=10)

resonator_drive = None

qubit_drive = GaussianPulse(
    frequency=4.020,
    amplitude=2.4 * np.sqrt(2),  # * np.linspace(0.75, 1.25, 10),
    phase=np.pi / 4,
    start_time=0,
    duration=16,
    sigma=4,
    drag_alpha=0.5 / 0.96542265872240614,  # 0.5 / 0.96542265872240614,
)

# Define the system
system = QubitResonatorSystem(
    qubit,
    resonator,
    coupling_strength=2 * np.pi * 0.250,
    qubit_pulse=qubit_drive,
    resonator_pulse=resonator_drive,
)


# Create experiment
ground_state = system.get_states(0, 0)
excited_state = system.get_states(1, 0)


experiment = SchroedingerExperiment(
    system,
    [ground_state],
    times,
    store_states=False,
    only_store_final=False,
    expectation_operators=[
        (
            system.qubit_state_occupation_operator(0)
            + system.qubit_state_occupation_operator(1)
        )
        / 2,
        (
            system.qubit_state_occupation_operator(0)
            + 1j * system.qubit_state_occupation_operator(1)
        )
        / 2,
        system.qubit_state_occupation_operator(1),
    ],
    save_path=experiment_path,
)


results = experiment.run()

analysis = automatic_analysis(results)


## Plotting ##

from matplotlib import pyplot as plt

fig, ax = plt.subplots(nrows=2, ncols=1)

# Pulse plot

pulse = [qubit_drive.pulse(t, None) for t in times]
ax[0].plot(times, pulse)


# Frequency plot
from numpy.fft import rfft, rfftfreq

freqs = rfftfreq(len(times), times[1] - times[0])
fft_pulse = rfft(pulse)

ax[1].plot(freqs, np.abs(fft_pulse))

omegas = qubit.hamiltonian.diag() / 2 / np.pi
shifts = dispersive_shift(system)

omega_01 = omegas[1] - omegas[0] + shifts[1] - shifts[0]
omega_12 = omegas[2] - omegas[1] + shifts[2] - shifts[1]

ax[1].axvline(omega_01, color="k", linestyle="--", label=r"$\omega_{01}$")
ax[1].axvline(omega_12, color="r", linestyle="--", label=r"$\omega_{12}$")


ax[0].set(
    xlabel="Time (ns)",
    ylabel="Amplitude",
    title="Pulse",
)
ax[1].set(
    xlabel="Frequency (GHz)",
    ylabel="Amplitude",
    title="Fourier Transform of Pulse",
    xlim=(3.5, 4.5),
)

fig.tight_layout()

print(results.exp_vals[..., -1])

H_0 = qubit.hamiltonian.full().reshape(4, 4, 1)
H_1 = (qubit.charge_matrix.full()).reshape(4, 4, 1) * pulse

H = H_0 + H_1

eigvals = np.array([np.linalg.eigvals(H[..., i]) for i in range(len(times))])

fig, ax = plt.subplots(nrows=2, ncols=1)

# ax[0].plot(times, eigvals[:, 0].real, label = r"$E_0$")
ax[0].plot(
    times,
    eigvals[:, 1].real / 1e3 / 2 / np.pi - eigvals[:, 0] / 1e3 / 2 / np.pi,
    label=r"$f_{01}$",
)
ax[0].set(
    ylabel="Frequency (MHz)",
    title="Eigenvalues of Hamiltonian",
)
ax[1].plot(
    times,
    eigvals[:, 2].real / 1e3 / 2 / np.pi - eigvals[:, 1] / 1e3 / 2 / np.pi,
    label=r"$f_{12}$",
)

ax[1].set(
    xlabel="Time (ns)",
    ylabel="Frequency (MHz)",
)
