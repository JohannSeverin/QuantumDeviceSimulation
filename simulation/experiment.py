import qutip 
import numpy as np
from qutip import mesolve, Options
from tqdm import tqdm


class SimulationExperiment:
    """
    Parent class for simulation of experiments. This class handles data management, sweeps and data storage.
    """
    def __init__(self, system, times, states, expectation_operators = None, store_states = False, only_store_final = False):
        # Load the variables into the class
        self.times  = times
        self.states = states

        # Find dimensions of the experiment
        self.number_of_states = len(states) if isinstance(states, list) else 1
        self.sweep_parameters = system.parameters_to_be_swept

        self.number_of_sweeps = sum([len(self.sweep_parameters[key]) for key in self.sweep_parameters.keys()])

        # Check if the number of sweeps is supported
        if self.number_of_sweeps not in [0, 1, 2]:
            raise NotImplementedError("Only 0, 1 or 2 sweeps are supported")
        
        # Output data handling
        self.expectation_operators = expectation_operators
        self.store_states          = store_states
        self.only_store_final      = only_store_final

        # Options to use for Simulation
        self.options = Options(
            store_states        = not self.only_store_final,
            store_final_state   = self.only_store_final,
        )

    def simulate(self, state):
        raise NotImplementedError("This method should be implemented in the subclass")

    def run(self):
        """
        Run all the experiments
        """
        if self.number_of_sweeps == 0:
            results = self.run_single_experiment()

        elif self.number_of_sweeps == 1:
            results = self.run_single_sweep()

        elif self.number_of_sweeps == 2:
            results = self.run_double_sweep()

        else:
            raise ValueError("Only 0, 1 or 2 sweeps are supported")
        
        return results

    def run_single_experiment(self):
        """
        Run a single experiment if no sweeps are required
        """
        results = {}

        if self.number_of_states == 1:
            state               = self.states if isinstance(self.states, qutip.Qobj) else self.states[0]
            simulation_result   = self.simulate(state)

            if self.only_store_final:
                simulation_result.states = simulation_result.states[-1]
            
            if self.store_states:
                results["states"]       = simulation_result.states

            if self.expectation_operators:
                results["exp_vals"]     = self.get_expvals(simulation_result.states, self.expectation_operators)
     
        else:

            # Create lists for the loop of states
            if self.store_states:
                results["states"]        = []

            if self.expectation_operators:
                results["exp_vals"]      = []

            for state in self.states:
                
                # Simulate
                simulation_result   = self.simulate(state)

                if self.only_store_final:
                    simulation_result.states = simulation_result.states[-1]

                # Store in dicts
                if self.store_states:
                    results["states"].append(simulation_result.states)

                if self.expectation_operators:
                    results["exp_vals"].append(self.exp_vals(simulation_result.states, self.expectation_operators))
                
            # Convert to numpy arrays
            if self.store_states:
                results["states"]       = np.array(results["states"])

            if self.expectation_operators:
                results["exp_vals"]     = np.array(results["exp_vals"])

            return results
                
            
        return results

    def run_single_sweep(self):
        """
        Sweeping over a single parameter
        """
        # sweep_param = [key + value for key, value in self.sweep_parameters.items() if value][0]
        
        for device, parameter in self.sweep_parameters.items():

            # Only one parameter in this instance
            sweep_param, sweep_device = parameter[0], device 

            if device == "system":
                sweep_list  = self.system.sweep_list[sweep_param]
            else:
                sweep_list = self.system.devices[device].sweep_list[sweep_param]
        
        # Create dict for results
        results = {
            "sweep_device": sweep_device,
            "sweep_param":  sweep_param,
            "sweep_list":   sweep_list,
        }

        # Create lists for the loop of states
        if self.store_states:
            results["states"]        = []

        if self.expectation_operators:
            results["exp_vals"]      = []

        system_params = self.system.parameters

        # Loop over values in sweep list
        for value in tqdm(sweep_list):

            # Update system parameters
            system_params[sweep_device][sweep_param] = value
            self.system.update(system_params)
            
            # Run single experiment with updated sweep param
            single_experiment_result = self.run_single_experiment()

            # Convert to numpy arrays
            if self.store_states:
                results["states"].append(single_experiment_result["states"])

            if self.expectation_operators:
                results["exp_vals"].append(single_experiment_result["exp_vals"])
        
        # Convert final to np-array
        if self.store_states:
            results["states"]   = np.array(results["states"])

        if self.expectation_operators:
            results["exp_vals"]     = np.array(results["exp_vals"])
        
        return results
    
    def run_double_sweep(self):
        sweep_param_0, sweep_param_1 = [key for key, value in self.sweep_parameter.items() if value]
        sweep_list_0, sweep_list_1   = self.pulse_arguments[sweep_param_0], self.pulse_arguments[sweep_param_1]

        # Create dict for results
        results = {
            "sweep_param": (sweep_param_0, sweep_param_1),
            "sweep_list": (sweep_list_0, sweep_list_1)
        }

        if self.final_state:
            results["final_state"]      = []
        if self.running_states:
            results["running_states"]   = []
        if self.running_exp_vals:
            results["exp_vals"]         = []
        
        data_points = len(sweep_list_0) * len(sweep_list_1)
        pbar = tqdm(total = int(data_points), desc = "Sweeping")
        pbar.update(0)
       
        for value_param_0 in sweep_list_0:

            results_sweep_param_0 = {key: [] for key in results.keys()}

            for value_param_1 in sweep_list_1:
                # Set new value
                self.pulse_arguments[sweep_param_0] = value_param_0
                self.pulse_arguments[sweep_param_1] = value_param_1

                # Run single experiment with updated sweep param
                single_experiment_result = self.run_single_experiment()
                
                # Store results
                if self.final_state:
                    results_sweep_param_0["final_state"].append(single_experiment_result["final_state"])
                if self.running_states:
                    results_sweep_param_0["running_states"].append(single_experiment_result["running_states"])
                if self.running_exp_vals:
                    results_sweep_param_0["exp_vals"].append(single_experiment_result["exp_vals"])

                pbar.update(1)

            # Propagate results to main dict
            for key in results.keys():
                if key in ["final_state", "running_states", "exp_vals"]:
                    results[key].append(np.array(results_sweep_param_0[key]))

        pbar.close()        
        return results

    def exp_vals(self, list_of_states, operators):
        """
        Get the expectation values of an operator for a list of states
        """
        if len(operators) > 1:
            list_of_expvals = []

        for operator in operators:
            if isinstance(operator, tuple):
                op, dimension_to_keep = operator
                states    = [state.ptrace(dimension_to_keep) for state in list_of_states]
            else:
                states      = list_of_states
                op          = operator

            exp_vals    = qutip.expect(op, states)

            if len(operators) > 1:
                list_of_expvals.append(exp_vals)
            else:
                return exp_vals
            
        return np.array(list_of_expvals)
    
             

class SchroedingerExperiment(SimulationExperiment):
    """
    Experiment for solving the Schrodinger equation for a quantum device system. 
    """

    def __init__(self, system, states, times, expectation_opreators = None, store_states = False, only_store_final = False):
        self.system = system
        
        super().__init__(system, times, states, expectation_operators = expectation_opreators, store_states = store_states, only_store_final = only_store_final)
    
    def simulate(self, state):
        """
        Simulate the system.
        """
        H_0 = self.system.hamiltonian

        if self.system.hamiltonian_t:
            H_1 = self.system.hamiltonian_t
            H   = [H_0, H_1]
        else:
            H = H_0

        return qutip.sesolve(
            H,
            psi0    = state,
            tlist   = self.times,
            options = self.options
        )



class LindbladExperiment(SimulationExperiment):
    """
    This simulation class also takes dissipation into account. It is based on the Lindblad master equation.
    """
    
    def __init__(self, system, states, times, expectation_opreators = None, store_states = False, only_store_final = False):
        raise NotImplementedError("This class is not yet implemented")
    

class StochasticExperiment(SimulationExperiment):
    """
    This simulation class also takes dissipation into account but also measurement backaction. 
    It is based on the stochastic master equation and can be combined with NTraj keyword to create multiple simulations
    as well 
    """
    
    def __init__(self, system, states, times, expectation_opreators = None, store_states = False, only_store_final = False):
        raise NotImplementedError("This class is not yet implemented")