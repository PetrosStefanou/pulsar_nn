
import numpy as np
import matplotlib.pyplot as plt


"""
   Module containing utility functions for plotting and analysis.
"""

def set_matplotlib_defaults(SMALL_SIZE = 20, MEDIUM_SIZE = 25, BIGGER_SIZE = 30):
   """
      Set default parameters for matlotlib
   """
   plt.rcParams['figure.constrained_layout.use'] = True
   plt.rcParams['figure.figsize'] = (16,10)

   plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
   plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
   plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
   plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
   plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
   plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
   plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title