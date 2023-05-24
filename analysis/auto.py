import matplotlib.pyplot as plt
import numpy as np
import os, sys

plt.style.use("../analysis/standard_plot_style.mplstyle")




class Analysis:
    """
    Parent class for handling analysis of data from experiments.
    """

    def __init(self):
            
        pass    



class BasicAuto(Analysis):

    def __init__(self, results, **kwargs):
        self.results            = results

        self.time_axis          = not results.only_store_final
        self.number_of_sweeps   = results.number_of_sweeps

        # We only plot expectation values in this class.
        if results.number_of_expvals == 0:
            return None

        super().__init__(**kwargs)

    def plot(self):
        # Old school match case for number of sweeps and the time 
        if self.time_axis:
            if self.number_of_sweeps == 0:
                print("Plotting expectation values over time without sweeps.")
                return self.plot_no_sweep_over_time(self.results)   
        
        else:
            if self.number_of_sweeps == 1:
                print("Plotting expectation values with 1 sweep.")
                return self.plot_final_one_sweep(self.results)

            elif self.number_of_sweeps == 2:
                print("Plotting expectation values with 2 sweeps.")
                return self.plot_final_two_sweeps(self.results)

        
        raise Warning(f"Basic Analysis does not support {self.number_of_sweeps} sweeps with time_axis = {self.time_axis}.")
        return None

    def plot_final_two_sweeps(self, results):
        exp_vals     = results.exp_vals
        sweep_lists  = results.sweep_lists
        sweep_params = results.sweep_params

        fig, axes = plt.subplots(ncols = results.number_of_states, nrows = results.number_of_expvals, sharex = True, sharey = True)
        
        axes = axes.reshape(results.number_of_expvals, results.number_of_states)

        if results.number_of_states  == 1:
            axes = np.expand_dims(axes, axis = -1)

        if results.number_of_expvals == 1:
            exp_vals = np.expand_dims(exp_vals, axis = -1)

        for i in range(results.number_of_expvals):
            for j in range(results.number_of_states):
                im = axes[i, j].imshow(exp_vals[..., j, i].T, extent = [sweep_lists[0].min(), sweep_lists[0].max(), sweep_lists[1].min(), sweep_lists[1].max()], 
                    aspect = "auto", origin = "lower", cmap = "plasma")
                axes[i, j].set(
                    xlabel = sweep_params[0][0],
                    ylabel = sweep_params[1][0],
                    title = f"State {j}"
                )
                fig.colorbar(im, ax = axes[i, j])

        return fig, axes

    def plot_final_one_sweep(self, results):
        exp_vals    = results.exp_vals
        sweep_list  = results.sweep_lists
        sweep_param = results.sweep_params[0][0]

        fig, axes = plt.subplots(ncols = results.number_of_states, nrows = results.number_of_expvals, sharex = True, sharey = True)

        if results.number_of_expvals == 1:
            for j in range(results.number_of_states):
                axes[j].plot(sweep_list, exp_vals[:, j])

                axes[j].set(
                    xlabel = sweep_param,
                    title = f"State {j}"
                )
            axes[0].set(ylabel = "exp_val")
        else:
            for i in range(results.number_of_expvals):
                for j in range(results.number_of_states):
                    axes[i, j].plot(sweep_list, exp_vals[:, j, i])

                    axes[i, j].set(
                        xlabel = sweep_param,
                        ylabel = "exp_val"
                    )
            for i in range(results.number_of_expvals):
                axes[i, 0].set(ylabel = "exp_val")
            for j in range(results.number_of_states):
                axes[0,  j].set(title = f"State {j}")
                axes[-1, j].set(xlabel = sweep_param)
        
        return fig, axes

    def plot_no_sweep_over_time(self, results):
        times       = results.times
        exp_vals    = results.exp_vals

        fig, axes = plt.subplots(ncols = results.number_of_states, nrows = results.number_of_expvals)

        if results.number_of_expvals == 1:
            for j in range(results.number_of_states):
                axes[j].plot(times, exp_vals[j, :])

                axes[j].set(
                    xlabel = "Time (ns)",
                    ylabel = "exp_val",
                    title = f"State {j}"
                )
        else:
            for i in range(results.number_of_expvals):
                for j in range(results.number_of_states):
                    axes[i, j].plot(times, exp_vals[i, j, :])

                    axes[i, j].set(
                        xlabel = "Time (ns)",
                        ylabel = "exp_val",
                        title = f"State {j}, exp_val {i}"
                    )

        fig.tight_layout()
        return fig, axes