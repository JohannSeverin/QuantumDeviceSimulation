import qutip 
import numpy as np

from qutip import mesolve, Options

from tqdm import tqdm


class Device:
    """
    Parent class for all devices. This is intended to take of common operations like sweeping parameters for an individual class and propagating them to the system.
    """

    def __init__(self):
        """
        Initialize the device.
        """
        if not hasattr(self, "parameters"):
            raise ValueError("The device must have a parameters attribute.")
        
        if not hasattr(self, "update_methods"):
            raise ValueError("The device must have an update_methods attribute.")

        self.parameters_to_be_swept()

        if self.should_be_swept:
            for key in self.sweep_parameters:
                self.parameters[key] = self.sweep_list[key][0]
        
        self.update(self.parameters)

    def parameters_to_be_swept(self):
        """
        Returns a list of the parameters that should be swept.
        """
        self.should_be_swept = np.any([isinstance(self.parameters[key], np.ndarray) for key in self.parameters.keys()])

        if self.should_be_swept:
            self.sweep_parameters = [key for key in self.parameters.keys() if isinstance(self.parameters[key], np.ndarray)]
            self.sweep_list       = {key: self.parameters[key] for key in self.sweep_parameters}
        else:
            self.sweep_parameters = None
            self.sweep_list       = None

    def update(self, new_parameters):
        """
        Update the parameters of the device.
        """
        self.parameters.update(new_parameters)

        for update_func in self.update_methods:
            update_func()

