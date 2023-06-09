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


from scipy.stats import norm


class GaussianPulse(Device):
    def __init__(
        self,
        frequency: float,
        amplitude: float,
        sigma: float,
        start_time=0,
        duration=0,
        phase=0,
        drag_alpha=0,
    ):
        # Set parameters
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.duration = duration
        self.sigma = sigma
        self.start_time = start_time
        self.drag_alpha = drag_alpha

        # To parent class
        self.sweepable_parameters = [
            "frequency",
            "amplitude",
            "phase",
            "duration",
            "start_time",
            "drag_alpha",
        ]
        self.update_methods = [self.set_pulse]

        super().__init__()

    def set_pulse(self) -> None:
        # envelope
        norm_dist = norm(loc=self.start_time + self.duration / 2, scale=self.sigma)
        self.envelope = lambda t: self.amplitude * norm_dist.pdf(t)

        cosine_t = lambda t: np.cos(self.phase) * np.cos(2 * np.pi * self.frequency * t)
        sine_t = lambda t: np.sin(self.phase) * np.sin(2 * np.pi * self.frequency * t)

        def pulse(t, _):
            if t > self.start_time and t < self.start_time + self.duration:
                envelope_at_t = self.envelope(t)
                pulse_I = envelope_at_t * cosine_t(t)
                pulse_Q = -self.drag_alpha * (
                    (t - (self.start_time + self.duration / 2))
                    / self.sigma**2
                    * envelope_at_t
                    * sine_t(t)
                )
                return pulse_I + pulse_Q
            else:
                return 0

        self.pulse = pulse


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
        def pulse(t, _):
            if t >= self.start_time and t < self.start_time + self.duration:
                envelope = self.amplitude * np.tanh(self.scale * (t - self.start_time))
                return envelope * np.cos(2 * np.pi * self.frequency * t + self.phase)
            else:
                return 0

        self.pulse = pulse
