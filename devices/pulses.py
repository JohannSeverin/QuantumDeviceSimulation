import qutip
import numpy as np

from qutip import tensor, basis

from devices.device import Device


################### Abstract Pulse Class ###################
class Pulse(Device):
    def __init__(self):
        """ """
        self.parameters = {}
        self.update_methods = []
        super().__init__()

    def set_pulse(self):
        raise NotImplementedError(
            "Pulse class is abstract. Implement set_pulse method."
        )


################### Pulses ###################
class SquareCosinePulse(Pulse):
    def __init__(self, frequency, amplitude, start_time=0, duration=None, phase=0):
        self.parameters = {
            "frequency": frequency,
            "amplitude": amplitude,
            "phase": phase,
            "duration": duration,
            "start_time": start_time,
        }

        self.update_methods = [self.set_pulse]

        super().__init__()

    def set_pulse(self):
        if self.parameters["duration"] is None:
            self.pulse = lambda t, _: self.parameters["amplitude"] * np.cos(
                2 * np.pi * self.parameters["frequency"] * t + self.parameters["phase"]
            )
        else:

            def pulse(t, _):
                if (
                    t >= self.parameters["start_time"]
                    and t < self.parameters["start_time"] + self.parameters["duration"]
                ):
                    return self.parameters["amplitude"] * np.cos(
                        2 * np.pi * self.parameters["frequency"] * t
                        + self.parameters["phase"]
                    )
                else:
                    return 0

            self.pulse = pulse
