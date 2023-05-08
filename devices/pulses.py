import qutip
import numpy as np

from qutip import tensor, basis


class ReadoutCosinePulse:

    def __init__(self, system, duration = None, frequency = None, amplitude = None, phase = 0):
        self.system = system
        self.frequency = frequency
        self.amplitude = amplitude
        self.duration = duration
        self.phase = phase

        self.intreaction_hamiltoinan = self.system.get_resonator_interaction()

    
    def get_time_dependent_hamiltonian(self):
        function_as_string = "A * cos(2 * pi * f * t + phi)"

        default_parameters = {
            "A": self.amplitude,
            "f": self.frequency,
            "phi": self.phase
        }

        return [self.intreaction_hamiltoinan, function_as_string], default_parameters








