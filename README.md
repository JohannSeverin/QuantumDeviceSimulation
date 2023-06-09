# QuantumDeviceSimulation
The goal of this project is to make a comprehentible way to simulate superconducting qubit readouts (and hopefully also control). As a beginning, the project will be focused on simulating the readout of a transmon qubit. 

# Current Functionality 
Currently, the following functionality is implemented:

## Device Classes
The following classes are implemented:
- **Transmon-Qubit** - Only a transmon qubit is implemented which takes the EJ, EC and ng parameters to return hamiltonian for the qubit with n-levels as well as the charge operator in the eigenbasis. 
- **Resonator** - A resonator class is implemented which takes the frequency and anharmonicity of the resonator to return the hamiltonian for the resonator with n-levels.
- **Resonator-Qubit-System** A simple system class which combines the two devices with a coupling strength to return the full hamiltonian of the system.
- **Pulse** A simple pulse class which takes the amplitude, frequency and duration of a cosine square pulse to return the pulse as a function of time.

## Experiment/Simulation Classes
The following classes are implemented:
- **Lindblad Experiment** is used as a playgound to make the general infrastructure for the experiment. This should pass sweepable quantities along to the pulse class to make a list of experiments to run for different initials states. The results are the stored. At the moment sweeps over 1 parameters is supported along with the ability to do each experiment for different initial states.

# Progress - Coding:
- [x] Create classes to model the devices
  - [ ] Qubit Class
    - [x] Transmon from device parameters
    - [ ] General Qubit from calibration parameters
  - [x] Resonator Class
  - [x] System Class (coupling the two together)
    - [x] Simpel system with qubit and resonator 
  - [x] Pulses
    - [x] Cosine Square Pulse
- [x] Create classes to model experiments using different simulation framework
  - [X] Simulation Class
    - [x] Handel input parameters
      - [x] single
      - [x] 1d sweep
      - [x] 2d sweep
    - [x] Handel output results
    - [x] Convert to parent class
    - [x] Proper output data structure
  - [x] Unitary (for simulating the Schrodinger equation)
  - [x] Lindblad Simulations (for simulating the master equation) 
  - [x] Monte Carlo Simulations (might speed up Lindblad simulation for some cases)
  - [ ] Stochastic Simulations (for simulating backaction and measurement)
- [x] Analysis Module - Possible in Seperate folder and compatible with OPX_control data
  - [x] Simple plotting
    - [x] Plotting of results
    - [x] Plotting of sweeps
  - [ ] IQ plots
- [ ] General Optimization
  - [ ] Enter reference frames in the System Class to simplify hamiltonian (maybe not possible in qutip since time dependence should be a scaler)
  - [ ] Dispersive approximation (should be written either as new system or wrapper for old system/pulse)
  - [ ] Write sweeps using multiprocessing
- [ ] Tidy up 
  - [ ] Go throgh units
  - [ ] Take care of cython files which accumulates in simulation folder
  - [x] Make module loading without having to add to path
  - [ ] Add documentation
  - [ ] Rewrite parts to be more like OPX_control
- [ ] Far-out Goals
  - [x] Single Qubit Control - to use for example with cloaking
  - [ ] Implement ways to use PyTorch to simluate/train the system
  - [ ] Multi Qubit Systems


# Next on List - Physics:
- [ ] Punchout Plot (amplitude - frequency scan)
  - [x] Implement 2d sweep
  - [ ] Some hours of of running code
- [ ] Zeno Effects
  - [ ] Stochastic Simulations
  - [ ] Qubit Control
- [ ] Qubit cloaking
  - [x] Single Qubit Control
  - [ ] More advanced readout pulses
- [ ] Machine Learning Readout
  - [ ] Stochastic Simulations