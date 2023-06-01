import sys
from typing import Any

sys.path.append("../")

import numpy as np
from abc import ABC, abstractmethod

from abc import ABC, abstractmethod
from dataclasses import dataclass


import qutip
from qutip import tensor, basis, Qobj

from devices.device import Device
from devices.pulses import Pulse


class DeviceManager:
    """
    Class for handling devices as dot-object while still allowing a dict like access.
    """

    def __init__(self, **kwargs):
        for key, values in kwargs.items():
            setattr(self, key, values)

    def __iter__(self):
        for key, value in self.__dict__.items():
            yield key, value


class System(ABC):
    """
    General system class.
    This handles primarily the parameters and sweep updates for the system.
    To use this class the child class must have a self.devices dictionary with the devices in the system.
    If any parameters on a system level should be swept, they must be added to the self.parameters dictionary.
    """

    sweepable_parameters: list = []
    update_methods: list = []
    devices: DeviceManager

    def __init__(self):
        self.system_parameters_to_be_swept()

        if "system" in self.sweep_parameters.keys():
            for param in self.sweep_parameters["system"]:
                setattr(self, param, self.sweep_parameters["system"][param][0])

        self.update()

    def update(self, new_parameters: dict = {}) -> None:
        """
        Update the parameters of the device.
        """
        for device_key in new_parameters:
            if device_key == "system":
                for parameter_key in new_parameters[device_key]:
                    self.set_parameter(
                        parameter_key, new_parameters[device_key][parameter_key]
                    )
            else:
                getattr(self.devices, device_key).update(new_parameters[device_key])

        # Update system parameters
        if self.should_be_swept:
            for key in self.sweep_parameters:
                if "system" in new_parameters.keys():
                    self.set_parameter(key, new_parameters[key])

        # Propagate the paramters to the devices
        for name, device in self.devices:
            if device:
                if device.should_be_swept and name in new_parameters.keys():
                    device.update(new_parameters[name])

        # Call the update methods
        for update_func in self.update_methods:
            update_func()

    def set_parameter(self, key: str, value: any) -> None:
        """
        Set a parameter of the device.

        Parameters
        ----
        key: str: The key of the parameter
        value: float: The value of the parameter
        """
        setattr(self, key, value)

    def get_parameter(self, key: str) -> any:
        """
        Get a parameter of the device.

        Parameters
        ----
        key: str: The key of the parameter
        """
        return getattr(self, key)

    def system_parameters_to_be_swept(self) -> None:
        """
        Returns a list of the parameters that should be swept.
        """
        # Find the parameters which have a list
        to_sweep = [
            key
            for key in self.sweepable_parameters
            if isinstance(self.get_parameter(key), np.ndarray)
        ]

        self.should_be_swept = len(to_sweep) > 0

        system_sweep_parameters = {key: self.get_parameter(key) for key in to_sweep}

        self.sweep_parameters = (
            {"system": system_sweep_parameters} if self.should_be_swept else {}
        )

        for device_key, device in self.devices:
            if device:
                if device.should_be_swept:
                    self.sweep_parameters[device_key] = device.sweep_parameters

    @abstractmethod
    def set_operators(self):
        pass

    @abstractmethod
    def get_states(self):
        pass


################### Qubit System Classes ###################
class QubitSystem(System):
    def __init__(
        self,
        qubit: Device,
        qubit_pulse: Pulse = None,
    ):
        """
        A class to create a simple qubit-resonator system.

        Parameters
        ----
        qubit: A qubit class, only Transmon exists at the moment
        resonator: a resonator class
        coupling: float, the strength of the coupling between qubit and resonator
        """
        self.devices = DeviceManager(
            qubit=qubit,
            qubit_pulse=qubit_pulse,
        )

        # Set methods to be updated
        self.update_methods = [self.set_operators, self.set_dissipators]

        # Set dimensions
        self.dimensions = {"qubit": qubit.levels}

        super().__init__()

    def set_dissipators(self):
        """
        Set the dissipators of the system.
        Currently the qubit and resonator are considered seperately.
        """
        qubit_dissipators = self.devices.qubit.dissipators
        self.dissipators = qubit_dissipators

    def set_operators(self):
        """
        Set operators. Basically just Hamiltonian object.
        """
        # devices
        qubit = self.devices.qubit

        # Time independent hamiltonian
        H_0_qubit = qubit.hamiltonian

        self.hamiltonian = [H_0_qubit]

        # Time dependent hamiltonian
        if self.devices.qubit_pulse:
            qubit_pulse = self.devices.qubit_pulse
            qubit_coupling_operator = qubit.charge_matrix

            self.hamiltonian.append([qubit_coupling_operator, qubit_pulse.pulse])

    def get_states(self, state: int = 0):
        return basis(self.dimensions["qubit"], state)

    def qubit_state_operator(self):
        """
        Operator to get expectation value of qubit state
        """
        return (qutip.num(self.devices.qubit.levels),)

    def qubit_state_occupation_operator(self, state: int = 1):
        """
        Operator to get expectation value of a qubit in a given state
        """
        return qutip.ket2dm(basis(self.devices.qubit.levels, state))


################### Qubit-Resonator System Classes ###################
class QubitResonatorSystem(System):
    """
    A class to create a simple qubit-resonator system.

    Parameters:
    ----
    qubit: A qubit class, only Transmon exists at the moment.
    resonator: a resonator class
    coupling: float, the strength of the coupling between qubit and resonator given in h * GHz
    reslator_pulse: A pulse class to interact with the resonator
    qubit_pulse: A pulse class to interact with the qubit
    """

    def __init__(
        self,
        qubit: Device,
        resonator: Device,
        coupling_strength: float,
        resonator_pulse: Pulse = None,
        qubit_pulse: Pulse = None,
    ):
        """
        A class to create a simple qubit-resonator system.

        Parameters
        ----
        qubit: A qubit class, only Transmon exists at the moment
        resonator: a resonator class
        coupling: float, the strength of the coupling between qubit and resonator
        """
        self.devices = DeviceManager(
            qubit=qubit,
            resonator=resonator,
            resonator_pulse=resonator_pulse,
            qubit_pulse=qubit_pulse,
        )

        self.coupling_strength = coupling_strength

        # Set parameters
        self.sweepable_parameters = ["coupling_strength"]

        # Set methods to be updated
        self.update_methods = [self.set_operators, self.set_dissipators]

        # Set dimensions
        self.dimensions = {"qubit": qubit.levels, "resonator": resonator.levels}

        super().__init__()

    def set_dissipators(self):
        """
        Set the dissipators of the system.
        Currently the qubit and resonator are considered seperately.
        """
        qubit_dissipators = self.devices.qubit.dissipators
        qubit_dissipators = [
            tensor(qubit_dissipators[i], qutip.qeye(self.devices["resonator"].levels))
            for i in range(len(qubit_dissipators))
        ]

        resonator_dissipators = self.devices.resonator.dissipators
        resonator_dissipators = [
            tensor(qutip.qeye(self.devices.qubit.levels), resonator_dissipators[i])
            for i in range(len(resonator_dissipators))
        ]

        self.dissipators = qubit_dissipators + resonator_dissipators

    def set_operators(self):
        """
        Set operators. Basically just Hamiltonian object.
        """
        # devices
        qubit = self.devices.qubit
        resonator = self.devices.resonator
        # system parameters
        coupling = self.coupling_strength

        # Time independent hamiltonian
        H_0_qubit = tensor(qubit.hamiltonian, qutip.qeye(resonator.levels))
        H_0_resonator = tensor(qutip.qeye(qubit.levels), resonator.hamiltonian)
        H_interaction = coupling * tensor(
            qubit.charge_matrix, resonator.a_dag + resonator.a
        )

        self.hamiltonian = [H_0_qubit + H_0_resonator + H_interaction]

        # Time dependent hamiltonian
        if self.devices.resonator_pulse:
            resonator_pulse = self.devices.resonator_pulse
            resonator_coupling_operator = tensor(
                qutip.qeye(qubit.levels), resonator.coupling_operator
            )

            self.hamiltonian.append(
                [resonator_coupling_operator, resonator_pulse.pulse]
            )

        if self.devices.qubit_pulse:
            qubit_pulse = self.devices.qubit_pulse
            qubit_coupling_operator = tensor(
                qubit.charge_matrix, qutip.qeye(resonator.levels)
            )

            self.hamiltonian.append([qubit_coupling_operator, qubit_pulse.pulse])

    def get_states(
        self, qubit_states=0, resonator_states=0
    ):  # TODO: make this without so many if statements
        # Only integers
        if isinstance(qubit_states, int) and isinstance(resonator_states, int):
            qubit = basis(self.devices.qubit.levels, qubit_states)
            resonator = basis(self.devices.resonator.levels, resonator_states)
            return tensor(qubit, resonator)

        # Integer and list of integers
        elif isinstance(qubit_states, int) and isinstance(resonator_states, list):
            qubit = basis(self.devices.qubit.levels, qubit_states)
            resonator_states = [
                basis(self.devices.resonator.levels, state)
                for state in resonator_states
            ]
            return [tensor(qubit, resonator) for resonator in resonator_states]

        # List of integers and integer
        elif isinstance(qubit_states, list) and isinstance(resonator_states, int):
            qubit_states = [
                basis(self.devices.qubit.levels, state) for state in qubit_states
            ]
            resonator = basis(self.devices.resonator.levels, resonator_states)
            return [tensor(qubit, resonator) for qubit in qubit_states]

        # List of integers
        elif len(qubit_states) == len(resonator_states):
            qubits_states = [
                basis(self.devices.qubit.levels, state) for state in qubit_states
            ]
            resonator_states = [
                basis(self.devices.resonator.levels, state)
                for state in resonator_states
            ]
            return [
                tensor(qubit, resonator)
                for qubit, resonator in zip(qubits_states, resonator_states)
            ]

    def photon_number_operator(self):
        """
        Operator to get expectation value of photon count
        """
        return tensor(
            qutip.qeye(self.devices.qubit.levels),
            self.devices.resonator.a_dag * self.devices.resonator.a,
        )

    def qubit_state_operator(self):
        """
        Operator to get expectation value of qubit state
        """
        return tensor(
            qutip.num(self.devices.qubit.levels),
            qutip.qeye(self.devices.resonator.levels),
        )

    def qubit_state_occupation_operator(self, state: int = 1):
        """
        Operator to get expectation value of a qubit in a given state
        """
        return tensor(
            qutip.ket2dm(basis(self.devices.qubit.levels, state)),
            qutip.qeye(self.devices.resonator.levels),
        )


def dispersive_shift(system: QubitResonatorSystem):
    qubit = system.devices.qubit
    frequency = 2 * np.pi * system.devices.resonator.frequency
    coupling = system.coupling_strength

    # Calculate dispersive shifts
    # Multi qubit shifts
    g_squared_matrix = coupling**2 * abs(qubit.charge_matrix.full()) ** 2

    omega_ij_matrix = np.expand_dims(qubit.hamiltonian.diag(), 1) - np.expand_dims(
        qubit.hamiltonian.diag(), 0
    )

    chi_matrix = g_squared_matrix * (
        1 / (omega_ij_matrix - frequency) + 1 / (omega_ij_matrix + frequency)
    )

    # The dispersive shifts
    dispersive_shifts = chi_matrix.sum(axis=1)

    return dispersive_shifts


class DispersiveQubitResonatorSystem(System):
    """
    Corresponding to QubitResonatorSystem but with the dispersive approximation.

    Instead of drive,
    """

    def __init__(
        self, qubit, resonator, drive_frequency, drive_amplitude, coupling_strength
    ):
        """
        Corresponding to QubitResonatorSystem but with the dispersive approximation.

        Instead of drive,
        """

        self.devices = DeviceManager(qubit=qubit, resonator=resonator)

        self.coupling_strength = coupling_strength
        self.drive_frequency = drive_frequency
        self.drive_amplitude = drive_amplitude

        # Set parameters
        self.sweepable_parameters = [
            "coupling_strength",
            "drive_amplitude",
            "drive_frequency",
        ]

        # Set methods to be updated
        self.update_methods = [self.set_operators, self.set_dissipators]

        # Set dimensions
        self.dimensions = {"qubit": qubit.levels, "resonator": resonator.levels}

        super().__init__()

    def set_dissipators(self):
        qubit_dissipators = self.devices.qubit.dissipators
        qubit_dissipators = [
            tensor(qubit_dissipators[i], qutip.qeye(self.devices.resonator.levels))
            for i in range(len(qubit_dissipators))
        ]

        resonator_dissipators = self.devices.resonator.dissipators
        resonator_dissipators = [
            tensor(qutip.qeye(self.devices.qubit.levels), resonator_dissipators[i])
            for i in range(len(resonator_dissipators))
        ]

        self.dissipators = qubit_dissipators + resonator_dissipators

    def set_operators(self):
        # devices
        qubit = self.devices.qubit
        resonator = self.devices.resonator

        # system parameters
        drive_frequency = self.drive_frequency
        drive_amplitude = self.drive_amplitude

        dispersive_shifts = self.dispersive_shift()

        # Time independent hamiltonian
        Omega = 2 * np.pi * (resonator.frequency - drive_frequency)
        H_0_resonator = Omega * tensor(
            qutip.qeye(qubit.levels), resonator.a_dag * resonator.a
        )

        dipsersive_operator = Qobj(np.diag(dispersive_shifts))
        H_int = tensor(dipsersive_operator, resonator.a_dag * resonator.a)

        H_drive = drive_amplitude * tensor(
            qutip.qeye(qubit.levels), resonator.a_dag + resonator.a
        )

        self.hamiltonian = H_0_resonator + H_int + H_drive

        self.hamiltonian_t = None

    def dispersive_shift(self):
        qubit = self.devices.qubit
        frequency = 2 * np.pi * self.devices.resonator.frequency
        coupling = self.coupling_strength

        # Calculate dispersive shifts
        # Multi qubit shifts
        g_squared_matrix = coupling**2 * abs(qubit.charge_matrix.full()) ** 2

        omega_ij_matrix = np.expand_dims(qubit.hamiltonian.diag(), 1) - np.expand_dims(
            qubit.hamiltonian.diag(), 0
        )

        chi_matrix = g_squared_matrix * (
            1 / (omega_ij_matrix - frequency) + 1 / (omega_ij_matrix + frequency)
        )

        # The dispersive shifts
        dispersive_shifts = chi_matrix.sum(axis=1)

        return dispersive_shifts

    def get_states(self, qubit_states=0, resonator_states=0):
        # Only integers
        if isinstance(qubit_states, int) and isinstance(resonator_states, int):
            qubit = basis(self.devices.qubit.levels, qubit_states)
            resonator = basis(self.devices.resonator.levels, resonator_states)
            return tensor(qubit, resonator)

        # Integer and list of integers
        elif isinstance(qubit_states, int) and isinstance(resonator_states, list):
            qubit = basis(self.devices.qubit.levels, qubit_states)
            resonator_states = [
                basis(self.devices.resonator.levels, state)
                for state in resonator_states
            ]
            return [tensor(qubit, resonator) for resonator in resonator_states]

        # List of integers and integer
        elif isinstance(qubit_states, list) and isinstance(resonator_states, int):
            qubit_states = [
                basis(self.devices.qubit.levels, state) for state in qubit_states
            ]
            resonator = basis(self.devices.resonator.levels, resonator_states)
            return [tensor(qubit, resonator) for qubit in qubit_states]

        # List of integers
        elif len(qubit_states) == len(resonator_states):
            qubits_states = [
                basis(self.devices.qubit.levels, state) for state in qubit_states
            ]
            resonator_states = [
                basis(self.devices.resonator.levels, state)
                for state in resonator_states
            ]
            return [
                tensor(qubit, resonator)
                for qubit, resonator in zip(qubits_states, resonator_states)
            ]

    def photon_number_operator(self):
        return tensor(
            qutip.qeye(self.devices.qubit.levels),
            self.devices.resonator.a_dag * self.devices.resonator.a,
        )

    def qubit_state_operator(self):
        return tensor(
            qutip.num(self.devices.qubit.levels),
            qutip.qeye(self.devices.resonator.levels),
        )


if __name__ == "__main__":
    sys.path.append("..")

    from devices.device import Transmon, Resonator

    qubit = Transmon(EC=7.5 / 25, EJ=7.5, T1=0, levels=3)
    resonator = Resonator(frequency=np.linspace(5.0, 7.0, 21), levels=10, kappa=0.1)

    system = DispersiveQubitResonatorSystem(
        qubit=qubit,
        resonator=resonator,
        coupling_strength=0.1 * np.linspace(0, 1, 10),
        drive_frequency=6.0,
        drive_amplitude=0.1,
    )
