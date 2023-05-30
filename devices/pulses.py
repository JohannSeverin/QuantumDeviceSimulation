import qutip
import numpy as np

from abc import abstractmethod

from qutip import tensor, basis

from devices.device import Device


class Pulse(Device):
    pulse: callable

    def __init__(self):
        super().__init__()

    @abstractmethod
    def set_pulse(self):
        pass


class SquareCosinePulse(Device):
    def __init__(self, frequency, amplitude, start_time=0, duration=None, phase=0):
        # Set parameters
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.duration = duration
        self.start_time = start_time

        # To parent class
        self.sweepable_parameters = [
            "frequency",
            "amplitude",
            "phase",
            "duration",
            "start_time",
        ]
        self.update_methods = [self.set_pulse]

        super().__init__()

    def set_pulse(self):
        if self.duration is None:
            self.pulse = lambda t, _: self.amplitude * np.cos(
                2 * np.pi * self.frequency * t + self.phase
            )
        else:

            def pulse(t, _):
                if t >= self.start_time and t < self.start_time + self.duration:
                    return self.amplitude * np.cos(
                        2 * np.pi * self.frequency * t + self.phase
                    )
                else:
                    return 0

            self.pulse = pulse
