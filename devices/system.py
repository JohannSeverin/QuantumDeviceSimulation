import numpy as np

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
