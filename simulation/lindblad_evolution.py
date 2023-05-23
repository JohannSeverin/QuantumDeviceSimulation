import qutip 
import numpy as np

from qutip import mesolve, Options

from tqdm import tqdm


def get_sweep_lists(dictionary_of_arguments):
    """
    Get the number of parameters to sweep and the number of sweeps from a dictionary of arguments
    """
    number_of_params = len(dictionary_of_arguments)

    sweep_parameter   = {key: isinstance(value, np.ndarray) for key, value in dictionary_of_arguments.items()}
    number_of_sweeps = sum(sweep_parameter.values())

    return number_of_params, number_of_sweeps, sweep_parameter


class LindbladExperiment:
    """
    Class to evolve a density matrix under a Lindblad master equation while sweeping different parameters
    """
    def __init__(self, system, pulse, times, states, running_exp_vals = None, final_exp_vals = None, final_state = False, running_states = False):
        """
        Create the base class for running a Lindblad evolution experiment. 
        We should be able to sweep parameters and get output as either the whole evolution, expectation values or simply the final state.

        Parameters: 
        ----
        system: A system object, see devices/basic_system.py, this should contain the Hamiltonian and collapse operators for the system
        pulse: A pulse object, see devices/pulses.py, this should contain the time-dependent Hamiltonian and be compatible with the system
        times: A list of times to run the simulation at
        states: A list of states to run the simulation for
        running_exp_vals: A list of expectation values to calculate during the simulation
        final_exp_vals: A list of expectation values to calculate at the end of the simulation
        final_state: A boolean, if true the final state will be saved
        running_states: A boolean, if true the state will be saved at each t. Memory intensive!
        """

        self.times = times
        self.states = states

        # Get time-independent and collapse operators from the system 
        self.time_independent_hamiltonian = system.get_hamiltonian()
        self.collapse_operators           = system.get_collapse_operators()

        # Get the time depence from the pulse object
        self.time_dependent_hamiltonian   = pulse.get_time_dependent_hamiltonian()[0]
        self.pulse_arguments              = pulse.get_time_dependent_hamiltonian()[1]

        # Get the experiments. 
        self.number_of_states = len(states) if isinstance(states, list) else 1
        self.number_of_params, self.number_of_sweeps, self.sweep_parameter = get_sweep_lists(self.pulse_arguments)

        # Check if the number of sweeps is supported
        if self.number_of_sweeps not in [0, 1, 2]:
            raise NotImplementedError("Only 0, 1 or 2 sweeps are supported")
        
        # Load the considered variables into the class
        self.running_exp_vals = running_exp_vals
        self.running_states = running_states
        
        self.final_exp_vals = final_exp_vals
        self.final_state = final_state

        if self.final_exp_vals is not None:
            raise NotImplementedError("Final expectation values are not yet supported")

        # Options to use for Simulation
        self.options = Options(
            store_states = self.running_states or self.running_exp_vals is not None,
            store_final_state = self.final_state or self.final_exp_vals is not None,
        )

    def run_experiments(self):
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
            simulation_result   = self.simulate_one_configuration(state, self.pulse_arguments)

            if self.final_state:
                results["final_state"]      = simulation_result.states[-1]
            
            if self.running_states:
                results["running_states"]   = simulation_result.states

            if self.running_exp_vals:
                results["exp_vals"]         = self.get_expvals(simulation_result.states, self.running_exp_vals)
        
        else:
            # Create lists for the loop of states
            if self.final_state:
                results["final_state"]      = []
            if self.running_states:
                results["running_states"]   = []
            if self.running_exp_vals:
                results["exp_vals"]         = []
            
            for state in self.states:
                # Run simulation and store results
                simulation_result = self.simulate_one_configuration(state, self.pulse_arguments)

                if self.final_state:
                    results["final_state"].append(simulation_result.states[-1])
                
                if self.running_states:
                    results["running_states"].append(simulation_result.states)

                if self.running_exp_vals:
                    results["exp_vals"].append(self.get_expvals(simulation_result.states, self.running_exp_vals))
            
        return results

    def run_single_sweep(self):
        """
        Sweeping over a single parameter
        """
        sweep_param = [key for key, value in self.sweep_parameter.items() if value][0]
        sweep_list  = self.pulse_arguments[sweep_param]

        # Create dict for results
        results = {
            "sweep_param": sweep_param,
            "sweep_list": sweep_list,
        }

        if self.final_state:
            results["final_state"]      = []
        if self.running_states:
            results["running_states"]   = []
        if self.running_exp_vals:
            results["exp_vals"]         = []

        # Loop over values in sweep list
        for value in tqdm(sweep_list):
            
            # Set new value
            self.pulse_arguments[sweep_param] = value

            # Run single experiment with updated sweep param
            single_experiment_result = self.run_single_experiment()

            # Store results
            if self.final_state:
                results["final_state"].append(single_experiment_result["final_state"])
            if self.running_states:
                results["running_states"].append(single_experiment_result["running_states"])
            if self.running_exp_vals:
                results["exp_vals"].append(single_experiment_result["exp_vals"])

        if self.final_state:
            results["final_state"]      = np.array(results["final_state"])
        if self.running_states:
            results["running_states"]   = np.array(results["running_states"])
        if self.running_exp_vals:
            results["exp_vals"]         = np.array(results["exp_vals"])


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



        raise NotImplementedError("Double sweeps are not yet supported")

    def simulate_one_configuration(self, state, args):
        """
        Simulate the evolution of the system in the interval given in times
        """
        if self.time_dependent_hamitonian is None:
            H = self.time_independent_hamiltonian
        elif isinstance(self.time_dependent_hamiltonian, list):
            H = [self.time_independent_hamiltonian] + self.time_dependent_hamiltonian

        
        results = mesolve(
            H = [self.time_independent_hamiltonian, self.time_dependent_hamiltonian],
            rho0 = state,
            tlist = self.times,
            c_ops = self.collapse_operators,
            args = args,
            options = self.options
        )

        return results

    def get_expvals(self, list_of_states, operators):
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

             
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.style.use("../analysis/standard_plot_style.mplstyle")

    experiment_name = "test"
    experiment_path = "/mnt/c/Users/johan/OneDrive/Skrivebord/QDS_data/dump"

    import sys, pickle, os 
    sys.path.append("..")
    
    from devices.resonator      import Resonator
    from devices.transmon       import Transmon
    from devices.pulses         import ReadoutCosinePulse, DispersiveReadoutCosinePulse
    from devices.basic_system   import QubitResonatorSystem, DispersiveQubitResonatorSystem

    qubit = Transmon().from_device_parameters(
        EC = 15 * 2 * np.pi / 25, 
        EJ = 15 * 2 * np.pi, 
        n_cutoff = 15, 
        ng       = 0.0, 
        levels   = 4
    )
    
    resonator = Resonator(
        6.02, 
        levels = 10
    )

    system = DispersiveQubitResonatorSystem(
        qubit = qubit, 
        resonator = resonator, 
        coupling_strength = 0.250
    )

    states = system.get_states(qubit_states = 0, resonator_states = 0)

    pulse = DispersiveReadoutCosinePulse(
        system,
        amplitude = 0.10,
        frequency = 6.00,
        phase     = 0.0
    )

    times = np.linspace(0, 50, 1000)

    experiment = LindbladExperiment(
        system = system,
        pulse = pulse,
        times = times,
        states = states,
        running_states = False,
        running_exp_vals = [system.photon_number_operator()],
    )

    results = experiment.run_experiments()

    with open(os.path.join(experiment_path, experiment_name + ".pkl"), "wb") as file:
        pickle.dump(results, file)

    plt.plot(times, results["exp_vals"])
    
    
