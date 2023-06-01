import sys

sys.path.append("..")
import qutip
import numpy as np
from qutip import mesolve, Options
from tqdm import tqdm

import pickle

from abc import ABC, abstractmethod

from collections import namedtuple

from dataclasses import dataclass

from devices.system import System
from devices.device import Device

from typing import Union, Iterable


@dataclass()
class SimulationResults:
    number_of_sweeps: int
    number_of_states: int
    number_of_expvals: int
    sweep_dict: dict[dict:list]
    times: list[float]
    store_states: bool
    only_store_final: bool

    ntrajectories: int = None
    save_path: str = None
    dimensions: int = None
    states: np.ndarray = None
    exp_vals: np.ndarray = None

    def __post_init__(self):
        self.sweep_devices = list(self.sweep_dict.keys())
        self.sweep_parameters = []
        for device in self.sweep_devices:
            for param in self.sweep_dict[device]:
                self.sweep_parameters.append((device, param))

        self.descriptions()

    def save(self):
        if self.save_path:
            with open(self.save_path, "wb") as f:
                pickle.dump(self, f)
        else:
            raise ValueError("No save path specified")

    def load(self):
        with open(self.save_path, "rb") as f:
            return pickle.load(f)

    def descriptions(self):
        """
        Create a description of the results.
        """
        self.state_descriptions = []
        self.exp_val_descriptions = []

        if self.store_states:
            state_descriptions = ["state_dim1", "state_dim2"]

            if self.number_of_sweeps > 1:
                state_descriptions.append(self.sweep_parameters[0])
                state_descriptions.append(self.sweep_parameters[1])

            elif self.number_of_sweeps == 1:
                state_descriptions.append(self.sweep_parameters[0])

            if self.number_of_states > 1:
                state_descriptions.append("initial_states")

            if self.ntrajectories:
                state_descriptions.append("trajectories")

            if not self.only_store_final:
                state_descriptions.append("time")

            self.state_descriptions = state_descriptions

        if self.number_of_expvals > 0:
            exp_val_descriptions = []

            if self.number_of_sweeps > 1:
                exp_val_descriptions.append(
                    f"{self.sweep_parameters[0][0]}/{self.sweep_parameters[0][1]}"
                )
                exp_val_descriptions.append(
                    f"{self.sweep_parameters[1][0]}/{self.sweep_parameters[1][1]}"
                )
            elif self.number_of_sweeps == 1:
                exp_val_descriptions.append(
                    f"{self.sweep_parameters[0][0]}/{self.sweep_parameters[0][1]}"
                )

            if self.number_of_states > 1:
                exp_val_descriptions.append("initial_states")

            if self.number_of_expvals > 1:
                exp_val_descriptions.append("exp_vals")

            if not self.only_store_final:
                exp_val_descriptions.append("time")

            self.exp_val_descriptions = exp_val_descriptions


class SimulationExperiment(ABC):
    """
    Parent class for simulation of experiments. This class handles data management, sweeps and data storage.
    """

    def __init__(
        self,
        system: System,
        times: Iterable,
        states: Union[qutip.Qobj, Iterable],
        expectation_operators: list[qutip.Qobj] = [],
        store_states: bool = False,
        only_store_final: bool = False,
        save_path: str = None,
    ):
        # Load the variables into the class
        self.times = times
        self.states = states

        # Find dimensions of the experiment
        self.number_of_states = len(states) if isinstance(states, list) else 1
        self.sweep_parameters = system.sweep_parameters

        self.number_of_sweeps = sum(
            [len(self.sweep_parameters[key]) for key in self.sweep_parameters]
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
            sweep_dict=self.sweep_parameters,
            times=self.times,
            store_states=self.store_states,
            only_store_final=self.only_store_final,
            save_path=self.save_path,
            dimensions=system.dimensions,
        )

    @abstractmethod
    def simulate(self, state) -> qutip.solver.Result:
        pass

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
        results = {}

        # Create lists for the loop of states
        if self.store_states:
            results["states"] = []

        if self.expectation_operators:
            results["exp_vals"] = []

        sweep_device = list(self.sweep_parameters.keys())[0]
        sweep_param = list(self.sweep_parameters[sweep_device])[0]
        sweep_list = self.sweep_parameters[sweep_device][sweep_param]

        for value in tqdm(sweep_list):
            self.system.update({sweep_device: {sweep_param: value}})

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
        list_of_sweeps = []
        sweep_tuple = namedtuple("sweep_tuple", ["device", "parameter", "sweep_list"])

        for device in self.sweep_parameters:
            for parameter in self.sweep_parameters[device]:
                sweep_list = self.sweep_parameters[device][parameter]
                list_of_sweeps.append(sweep_tuple(device, parameter, sweep_list))

        outer, inner = list_of_sweeps

        results = {}

        # Create lists for the loop of states
        if self.store_states:
            results["states"] = []

        if self.expectation_operators:
            results["exp_vals"] = []

        pbar = tqdm(total=len(outer.sweep_list) * len(inner.sweep_list), leave=True)

        # Loop outer sweep
        for value_outer in outer.sweep_list:
            update_dict = {outer.device: {outer.parameter: value_outer}}

            inner_sweep_results = {
                key: [] for key in results if key in ["states", "exp_vals"]
            }

            # Loop inner sweep
            for value_inner in inner.sweep_list:
                if inner.device == outer.device:
                    update_dict[inner.device][inner.parameter] = value_inner
                else:
                    update_dict[inner.device] = {inner.parameter: value_inner}

                # Update system
                self.system.update(update_dict)

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
        self.results.save()


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


class MonteCarloExperiment(SimulationExperiment):
    """
    This simulation class also takes dissipation into account but also measurement backaction.
    It is based on the stochastic master equation and can be combined with NTraj keyword to create multiple simulations
    as well
    """

    def __init__(
        self,
        system: System,
        states: list[qutip.Qobj],
        times: list[float],
        ntraj: int = 1,
        exp_val_method="average",
        **kwargs,
    ):
        self.system = system
        self.ntraj = ntraj
        self.exp_val_method = exp_val_method

        super().__init__(system, times, states, **kwargs)

    def simulate(self, state):
        """
        Simulate the system.
        """
        H = self.system.hamiltonian

        return qutip.mcsolve(
            H,
            rho0=state,
            tlist=self.times,
            c_ops=self.system.dissipators,
            options=self.options,
            ntraj=self.ntraj,
        )

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
