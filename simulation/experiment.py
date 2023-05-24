import qutip
import numpy as np
from qutip import mesolve, Options
from tqdm import tqdm

import pickle


class SimulationExperiment:
    """
    Parent class for simulation of experiments. This class handles data management, sweeps and data storage.
    """

    def __init__(
        self,
        system,
        times,
        states,
        expectation_operators=None,
        store_states=False,
        only_store_final=False,
        save_path=None,
    ):
        # Load the variables into the class
        self.times = times
        self.states = states

        # Find dimensions of the experiment
        self.number_of_states = len(states) if isinstance(states, list) else 1
        self.sweep_parameters = system.parameters_to_be_swept

        self.number_of_sweeps = sum(
            [len(self.sweep_parameters[key]) for key in self.sweep_parameters.keys()]
        )

        # Check if the number of sweeps is supported
        if self.number_of_sweeps not in [0, 1, 2]:
            raise NotImplementedError("Only 0, 1 or 2 sweeps are supported")

        # Output data handling
        self.expectation_operators = expectation_operators
        self.store_states = store_states
        self.only_store_final = only_store_final

        # Options to use for Simulation
        self.options = Options(
            store_states=not self.only_store_final,
            store_final_state=self.only_store_final,
        )

        self.save_path = save_path

        self.results = SimulationResults(
            number_of_sweeps=self.number_of_sweeps,
            number_of_states=self.number_of_states,
            number_of_expvals=len(self.expectation_operators),
            sweep_params=list(self.sweep_parameters.values()),
            times=self.times,
            store_states=self.store_states,
            only_store_final=self.only_store_final,
            save_path=self.save_path,
        )

        self.results.dimensions = system.dimensions

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

        if self.save_path:
            self.save_results(results)

        if self.store_states:
            self.results.states = results["states"]
        if self.expectation_operators:
            self.results.exp_vals = results["exp_vals"]

        return self.results

    def run_single_experiment(self):
        """
        Run a single experiment if no sweeps are required
        """
        results = {}

        if self.number_of_states == 1:
            state = (
                self.states if isinstance(self.states, qutip.Qobj) else self.states[0]
            )
            simulation_result = self.simulate(state)

            if self.only_store_final:
                simulation_result.states = simulation_result.states[-1]

            if self.store_states:
                results["states"] = [simulation_result.states]

            if self.expectation_operators:
                results["exp_vals"] = self.exp_vals(
                    simulation_result.states, self.expectation_operators
                )

        else:
            # Create lists for the loop of states
            if self.store_states:
                results["states"] = []

            if self.expectation_operators:
                results["exp_vals"] = []

            for state in self.states:
                # Simulate
                simulation_result = self.simulate(state)

                if self.only_store_final:
                    simulation_result.states = simulation_result.states[-1]

                # Store in dicts
                if self.store_states:
                    results["states"].append(simulation_result.states)

                if self.expectation_operators:
                    results["exp_vals"].append(
                        self.exp_vals(
                            simulation_result.states, self.expectation_operators
                        )
                    )

            # Convert to numpy arrays
            if self.store_states:
                results["states"] = np.array(results["states"])

            if self.expectation_operators:
                results["exp_vals"] = np.array(results["exp_vals"])

            return results

        return results

    def run_single_sweep(self):
        """
        Sweeping over a single parameter
        """

        for device, parameter in self.sweep_parameters.items():
            # Only one parameter in this instance
            sweep_param, sweep_device = parameter[0], device

            if device == "system":
                sweep_list = self.system.sweep_list[sweep_param]
            else:
                sweep_list = self.system.devices[device].sweep_list[sweep_param]

            self.results.sweep_lists = sweep_list

        # Create dict for results
        results = {
            "sweep_device": sweep_device,
            "sweep_param": sweep_param,
            "sweep_list": sweep_list,
        }

        # Create lists for the loop of states
        if self.store_states:
            results["states"] = []

        if self.expectation_operators:
            results["exp_vals"] = []

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
            results["states"] = np.array(results["states"])

        if self.expectation_operators:
            results["exp_vals"] = np.array(results["exp_vals"])

        return results

    def run_double_sweep(self):
        """
        Sweeping over a single parameter
        """

        # Split to tuple with: (device_key, parameter_name, sweep_list)
        # Combine into list.
        sweep_tuples = []
        for device, list_of_parameters in self.sweep_parameters.items():
            for parameter in list_of_parameters:
                if device == "system":
                    sweep_list = self.system.sweep_list[parameter]
                else:
                    sweep_list = self.system.devices[device].sweep_list[parameter]

                sweep_tuples.append((device, parameter, sweep_list))

        results = {
            "sweep_device": [t[0] for t in sweep_tuples],
            "sweep_param": [t[1] for t in sweep_tuples],
            "sweep_list": [t[2] for t in sweep_tuples],
        }

        self.results.sweep_lists = [t[2] for t in sweep_tuples]

        outer, inner = sweep_tuples

        # Create lists for the loop of states
        if self.store_states:
            results["states"] = []

        if self.expectation_operators:
            results["exp_vals"] = []

        system_params = self.system.parameters

        pbar = tqdm(total=len(outer[2]) * len(inner[2]), leave=True)

        # Loop outer sweep
        for value_outer in outer[2]:
            system_params[outer[0]][outer[1]] = value_outer

            inner_sweep_results = {
                key: [] for key in results if key in ["states", "exp_vals"]
            }

            # Loop inner sweep
            for value_inner in inner[2]:
                system_params[inner[0]][inner[1]] = value_inner

                # Update system
                self.system.update(system_params)

                # Run experiment
                single_experiment_result = self.run_single_experiment()

                # Store results
                for key in inner_sweep_results:
                    inner_sweep_results[key].append(single_experiment_result[key])

                pbar.update(1)

            # Propagate results to outer loop
            for key in inner_sweep_results:
                results[key].append(np.array(inner_sweep_results[key]))

        # Convert to numpy arrays
        for key in results:
            if key in ["states", "exp_vals"]:
                results[key] = np.array(results[key])

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
                states = [state.ptrace(dimension_to_keep) for state in list_of_states]
            else:
                states = list_of_states
                op = operator

            exp_vals = qutip.expect(op, states)

            if len(operators) > 1:
                list_of_expvals.append(exp_vals)
            else:
                return exp_vals

        return np.array(list_of_expvals)

    def save_results(self, results):
        """
        Save results to file. Right now just pickling the results dict.
        """
        with open(self.save_path, "wb") as f:
            pickle.dump(results, f)


class SimulationResults:
    """
    Container for results from a simulation experiment.
    """

    def __init__(
        self,
        number_of_sweeps,
        number_of_states,
        sweep_lists=None,
        sweep_params=None,
        store_states=False,
        number_of_expvals=False,
        only_store_final=False,
        save_path=None,
        **kwargs
    ):
        # Store meta stats for defining the result container
        self.number_of_sweeps = number_of_sweeps
        self.number_of_states = number_of_states
        self.number_of_expvals = number_of_expvals
        self.sweep_params = sweep_params
        self.sweep_lists = sweep_lists
        self.store_states = store_states
        self.only_store_final = only_store_final
        self.save_path = save_path
        self.state_dimensions = {}

        # Set additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Create empty lists for storing the results
        self.states = np.array([])
        self.exp_vals = np.array([])

        # Create list for describing the dimensions of the results
        self.create_descriptions()

    def create_descriptions(self):
        """
        Create a description of the results.
        """
        if self.store_states:
            state_descriptions = ["state_dim1", "state_dim2"]
            sweep_params = np.array(self.sweep_params).flatten()

            if self.number_of_sweeps > 1:
                state_descriptions.append(sweep_params[0])
                state_descriptions.append(sweep_params[1])

            elif self.number_of_sweeps == 1:
                state_descriptions.append(sweep_params[0])

            if self.number_of_states > 1:
                state_descriptions.append("initial_states")

            if not self.only_store_final:
                state_descriptions.append("time")

            self.state_descriptions = state_descriptions

        if self.number_of_expvals > 0:
            exp_val_descriptions = []
            sweep_params = np.array(self.sweep_params).flatten()

            if self.number_of_sweeps > 1:
                exp_val_descriptions.append(sweep_params[0])
                exp_val_descriptions.append(sweep_params[1])
            elif self.number_of_sweeps == 1:
                exp_val_descriptions.append(sweep_params[0])

            if self.number_of_states > 1:
                exp_val_descriptions.append("initial_states")

            if not self.only_store_final:
                exp_val_descriptions.append("time")

            if self.number_of_expvals > 1:
                exp_val_descriptions.append("exp_vals")

            self.exp_val_descriptions = exp_val_descriptions


class SchroedingerExperiment(SimulationExperiment):
    """
    Experiment for solving the Schroedinger equation for a quantum device system.
    """

    def __init__(self, system, states, times, **kwargs):
        self.system = system

        super().__init__(system, times, states, **kwargs)

    def simulate(self, state):
        """
        Simulate the system.
        """
        H = self.system.hamiltonian
        return qutip.sesolve(H, psi0=state, tlist=self.times, options=self.options)


class LindbladExperiment(SimulationExperiment):
    """
    This simulation class also takes dissipation into account. It is based on the Lindblad master equation.
    """

    def __init__(self, system, states, times, **kwargs):
        self.system = system

        super().__init__(system, times, states, **kwargs)

    def simulate(self, state):
        """
        Simulate the system.
        """
        H = self.system.hamiltonian

        return qutip.mesolve(
            H,
            rho0=state,
            tlist=self.times,
            c_ops=self.system.dissipators,
            options=self.options,
        )


class StochasticExperiment(SimulationExperiment):
    """
    This simulation class also takes dissipation into account but also measurement backaction.
    It is based on the stochastic master equation and can be combined with NTraj keyword to create multiple simulations
    as well
    """

    def __init__(
        self,
        system,
        states,
        times,
        expectation_opreators=None,
        store_states=False,
        only_store_final=False,
    ):
        raise NotImplementedError("This class is not yet implemented")
