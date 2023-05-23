import numpy as np

import qutip
from qutip import tensor, basis

class System:
    """
    General system class.
    This handles primarily the parameters and sweep updates for the system.
    To use this class the child class must have a self.devices dictionary with the devices in the system.
    If any parameters on a system level should be swept, they must be added to the self.parameters dictionary.
    """
    def __init__(self):        
        
        # Check if the syetem has parameters
        if hasattr(self, "parameters"):
            self.parameters = {}
        
        # Control that the system has devices
        if not hasattr(self, "devices"):
            raise AttributeError("System has no devices")
        
        # Update the dictionaries of the system to handle the substructure of the system
        self.parameters = {"system": self.system_parameters}
        
        # Add device parameters to the system parameters
        device_params = {key: self.devices[key].parameters for key in self.devices.keys()}
        self.parameters.update(device_params)
        
        # Check if any parameters should be swept
        self.system_parameters_to_be_swept()

        # List of devices to be swept
        if self.should_be_swept:
            self.parameters_to_be_swept = {"system": self.sweep_parameters}
        else:
            self.parameters_to_be_swept = {}
        
        for key in self.devices.keys():
            if self.devices[key].should_be_swept:
                self.parameters_to_be_swept[key] = self.devices[key].sweep_parameters
        
        
        # Set the parameters to the first parameter if they should be swept
        if self.should_be_swept:
            for key in self.sweep_parameters:
                self.parameters["system"][key] = self.sweep_list[key][0]
        
        # update using methods and first parameters
        self.update(self.parameters)
    
    def system_parameters_to_be_swept(self):
        """
        Returns a list of the parameters that should be swept.
        """
        self.should_be_swept = np.any([isinstance(self.system_parameters[key], np.ndarray) for key in self.system_parameters.keys()])

        if self.should_be_swept:
            self.sweep_parameters = [key for key in self.system_parameters.keys() if isinstance(self.system_parameters[key], np.ndarray)]
            self.sweep_list       = {key: self.system_parameters[key] for key in self.sweep_parameters}
        else:
            self.sweep_parameters = None
            self.sweep_list       = None

    def update(self, new_parameters):
        """
        Update the parameters of the device.
        """
        self.parameters.update(new_parameters)

        for device_key, device in self.devices.items():
            if device.should_be_swept:
                device.update(self.parameters[device_key])

        for update_func in self.update_methods:
            update_func()

class QubitResonatorSystem(System):
    """
    A class to create a simple qubit-resonator system.

    Parameters:
    ----
    qubit: A qubit class, only Transmon exists at the moment. 
    resonator: a resonator class
    coupling: float, the strength of the coupling between qubit and resonator given in h * GHz
    """
    def __init__(self, qubit, resonator, resonator_pulse, coupling_strength):
        """
        A class to create a simple qubit-resonator system.

        Parameters
        ----
        qubit: A qubit class, only Transmon exists at the moment
        resonator: a resonator class
        coupling: float, the strength of the coupling between qubit and resonator
        """
        self.devices = {
            "qubit": qubit,
            "resonator": resonator,
            "resonator_pulse": resonator_pulse
        }

        # Set parameters
        self.system_parameters = {"coupling": coupling_strength * 2 * np.pi}

        # Set methods to be updated
        self.update_methods = [self.set_opreators]

        super().__init__()

    def set_opreators(self):
        # devices
        qubit           = self.devices["qubit"]
        resonator       = self.devices["resonator"]
        resonator_pulse = self.devices["resonator_pulse"]
        
        # system parameters
        coupling = self.parameters["system"]["coupling"]

        # Time independent hamiltonian
        H_0_qubit       = tensor(qubit.hamiltonian, qutip.qeye(resonator.levels))
        H_0_resonator   = tensor(qutip.qeye(qubit.levels), resonator.hamiltonian)
        H_interaction   = coupling * tensor(qubit.charge_matrix, resonator.a_dag + resonator.a)

        self.hamiltonian = H_0_qubit + H_0_resonator + H_interaction

        # Time dependent hamiltonian
        resonator_coupling_operator = tensor(qutip.qeye(qubit.levels), resonator.coupling_operator)

        self.hamiltonian_t = [resonator_coupling_operator, resonator_pulse.pulse] 

    def get_states(self, qubit_states = 0, resonator_states = 0):
            # Only integers
            if isinstance(qubit_states, int) and isinstance(resonator_states, int):        
                qubit       = basis(self.devices["qubit"].levels, qubit_states)
                resonator   = basis(self.devices["resonator"].levels, resonator_states)
                return tensor(qubit, resonator)
            
            # Integer and list of integers
            elif isinstance(qubit_states, int) and isinstance(resonator_states, list):
                qubit       = basis(self.devices["qubit"].levels, qubit_states)
                resonator_states = [basis(self.devices["resonator"].levels, state) for state in resonator_states]
                return [tensor(qubit, resonator) for resonator in resonator_states]
            
            # List of integers and integer
            elif isinstance(qubit_states, list) and isinstance(resonator_states, int):
                qubit_states = [basis(self.devices["qubit"].levels, state) for state in qubit_states]
                resonator   = basis(self.devices["resonator"].levels, resonator_states)
                return [tensor(qubit, resonator) for qubit in qubit_states]

            # List of integers
            elif len(qubit_states) == len(resonator_states):
                qubits_states = [basis(self.devices["qubit"].levels, state) for state in qubit_states]
                resonator_states = [basis(self.devices["resonator"].levels, state) for state in resonator_states]
                return [tensor(qubit, resonator) for qubit, resonator in zip(qubits_states, resonator_states)]

    def photon_number_operator(self):
        return tensor(qutip.qeye(self.devices["qubit"].levels), self.devices["resonator"].a_dag * self.devices["resonator"].a)
    
    def qubit_state_operator(self):
        return tensor(qutip.num(self.devices["qubit"].levels), qutip.qeye(self.devices["resonator"].levels))


