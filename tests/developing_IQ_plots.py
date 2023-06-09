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


class CloakedPulse(Device):
    def __init__(
        self, frequency, amplitude, scale=1, start_time=0, duration=None, phase=0
    ):
        # Set parameters
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.duration = duration
        self.start_time = start_time
        self.scale = scale

        # To parent class
        self.sweepable_parameters = [
            "frequency",
            "amplitude",
            "phase",
            "duration",
            "start_time",
            "scale",
        ]
        self.update_methods = [self.set_pulse]

        super().__init__()

    def set_pulse(self):
        self.envelope = lambda t: self.amplitude * np.tanh(
            self.scale * (t - self.start_time)
        )

        def pulse(t, _):
            if t >= self.start_time and t <= self.start_time + self.duration:
                return self.envelope(t) * np.cos(
                    2 * np.pi * self.frequency * t + self.phase
                )
            else:
                return 0

        self.pulse = pulse


qubit_pulse = CloakedPulse(
    frequency=6.0,
    amplitude=-1 * coupling_strength * 1.02991037 * np.array([0, 1, 2]),
    scale=0.02187893,
    duration=100,
    start_time=0,
    phase=np.pi / 2,
)

resonator_pulse = SquareCosinePulse(
    amplitude=0.1,
    frequency=6.0,
    phase=0,
    duration=500,
    start_time=0,
)

## Define System
system = QubitResonatorSystem(
    qubit=qubit,
    resonator=resonator,
    qubit_pulse=qubit_pulse,
    resonator_pulse=resonator_pulse,
    coupling_strength=coupling_strength * np.array([0, 1]),
)

## Simulate
from simulation.experiment import (
    LindbladExperiment,
    MonteCarloExperiment,
    SchroedingerExperiment,
)

times = np.linspace(0, 100, 1000)

experiment = LindbladExperiment(
    system=system,
    states=[system.get_states(0, 0), system.get_states(1, 0)],
    times=times,
    store_states=True,
    only_store_final=False,
    expectation_operators=[
        system.photon_number_operator(),
        system.resonator_I(),
        system.resonator_Q(),
        system.qubit_state_occupation_operator(1),
    ],
    # ntraj=10,
    # exp_val_method="average",
)

results = experiment.run()


from analysis.auto import automatic_analysis

from analysis.Q_func import qfunc_plotter, qfunc_plotter_with_time_slider

automatic_analysis(results)
qfunc_plotter_with_time_slider(results, interval=2.5, demod_frequency=6.00)


fig, ax = plt.subplots(1, 2)
fig.suptitle("Drive on Resonator Frequency")

envelope = lambda t: 1.02991037 * np.tanh(
    qubit_pulse.scale * (t - qubit_pulse.start_time)
)

ax[0].plot(results.times, results.exp_vals[0, 0, 0, 0, :], "k--", label="No coupling")
ax[0].plot(results.times, envelope(results.times), label="Cloak Envelope")

ax[0].legend(fontsize=12, loc="upper left")

ax[1].plot(
    results.times,
    results.exp_vals[0, 0, 0, 0, :],
    label="No coupling",
    ls="--",
    color="k",
)

for i, strength in enumerate(results.sweep_dict["qubit_pulse"]["amplitude"]):
    relative_strength = strength / coupling_strength
    ax[1].plot(
        results.times,
        results.exp_vals[1, i, 0, 0, :],
        label=f"Cloak drive - strength: {int(abs(relative_strength)):}",
    )

ax[1].legend(fontsize=12, loc="upper left")
