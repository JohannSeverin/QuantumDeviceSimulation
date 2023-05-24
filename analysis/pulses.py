import matplotlib.pyplot as plt
import numpy as np
import os, sys

plt.style.use("../analysis/standard_plot_style.mplstyle")

sys.path.append("../")
from devices.pulses import SquareCosinePulse  # TODO - create pulse class.
