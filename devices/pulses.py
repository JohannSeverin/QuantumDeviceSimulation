import qutip
import numpy as np

from qutip import tensor, basis

from devices.device import Device


class ReadoutCosinePulse(Device):
    def __init__(self, frequency, amplitude, phase=0):
        self.parameters = {
            "frequency": frequency,
            "amplitude": amplitude,
            "phase": phase,
        }

        self.update_methods = [self.set_pulse]

        super().__init__()

    def set_pulse(self):
        self.pulse = lambda t, args: self.parameters["amplitude"] * np.cos(
            2 * np.pi * self.parameters["frequency"] * t + self.parameters["phase"]
        )
