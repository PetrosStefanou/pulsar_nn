
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, patches, colors
import tensorflow as tf



"""
   Module containing utility functions for plotting and analysis of the axisymmetric pulsar magnetosphere problem
   solved using Physics Informed Neural Networks.
"""

def set_matplotlib_defaults(SMALL_SIZE = 20, MEDIUM_SIZE = 25, BIGGER_SIZE = 30):
   """
      Set default parameters for matlotlib
   """
   plt.rcParams['figure.constrained_layout.use'] = True
   plt.rcParams['figure.figsize'] = (16,10)
   # plt.rcParams["text.latex.preamble"].append(r'\usepackage{xfrac}')

   plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
   plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
   plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
   plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
   plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
   plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
   plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title

   return

# ===================================================================================================

def f_b (X, P_c, r_c):

   q = X [:,0]
   mu = X [:,1]

   fb = (1 - mu ** 2) * (P_c + q * ((1 - P_c) * tf.nn.relu(1 - 1 / (q * r_c)) ** 2 / (1 - 1 / r_c) ** 2))

   return fb

# ===================================================================================================

def h_b (X, r_c):

   q = X [:,0]
   mu = X [:,1]

   hb = (1 - mu ** 2) * (1 - q) * tf.sqrt(tf.nn.relu (1 - 1 / (q * r_c)) ** 2 + mu ** 2)

   return hb

# ===================================================================================================

def P_parametrization (X, N_p, parameters):
   
   """
      Calculate P and P_c using the output of the network and the parametrisation
   """
   
   Np = N_p (X)
   r_c = parameters ['r_c']
   
   if parameters['Pc_mode'] == 'free':
      P_c = tf.reduce_mean (Np[:,1])
   elif parameters['Pc_mode'] == 'fixed':
      P_c = parameters['Pc_fixed']
   else:
      raise ValueError ("Pc_mode must be one of 'free', 'fixed'.\n\
                        Current value --> {}".parameters['Pc_mode'])

   P = f_b (X, P_c, r_c) + h_b (X, r_c) * Np[:,0]

   return P, P_c

# ===================================================================================================

def T_parametrization (P, P_c, N_t, R_lc):
   
   # Reshape N_t output from (n_points,1) to (n_points, ) for proper broadcasting
   N_t = tf.reshape(N_t (P), N_t (P).shape[0])

   T = (-2 * P / R_lc + P ** 2 * N_t) * tf.experimental.numpy.heaviside(P_c ** 2 - P ** 2, 0.5)
    
   return T


# ===================================================================================================


def plot_pinn(X, Z, P, Pc, fb, hb, Np, parameters):
   
   '''
      Plot the three functions that define the parametrisation (fb, hb and Np).
   '''
   
   fig, ax = plt.subplots(2,2)
   if parameters ['coordinates'] == 'spherical':
      slc = np.s_[:,1:-1]
   else:
      slc = np.s_[:,:]

   levels = 500
   color_star = '#CECFD2'
   color_lc = 'orange'
   color_sep = 'red'
   cmap = 'winter'
   min_value = np.floor(np.log10(min([i for i in [hb[hb > 0].min(), fb[fb > 0].min(), np.abs(Np[Np < 0]).min()]])))
   max_value = np.ceil(np.log10(max([i for i in [hb[hb > 0].max(), fb[fb > 0].max(), np.abs(Np[Np < 0]).max()]])))
   levels = np.logspace(min_value, max_value, 50)
   # levels = 100
   titles = ['$f_b$', '$h_b$', '$N_P$', '$\\frac{h_b N_P}{f_b}$']


   cont1 = ax[0,0].contourf (X[slc], Z[slc]
                           , fb[slc]
                           , levels=levels
                           , cmap=cmap
                           , norm='log'
                           )

   cont2 = ax[0,1].contourf (X[slc], Z[slc]
                           , hb[slc]
                           , levels=levels
                           , cmap=cmap
                           , norm='log'
                           )

   cont3 = ax[1,0].contourf (X[slc], Z[slc]
                           , np.abs(Np[slc])
                           , levels=levels
                           , cmap=cmap
                           , norm='log' 
                           )

   cont4 = ax[1,1].contourf (X[slc], Z[slc]
                           , hb[slc] * np.abs(Np[slc] / fb[slc])
                           , levels=levels
                           , cmap=cmap
                           , norm='log'
                           )

   for i, axis in enumerate(ax.ravel()):
      axis.set_aspect ('equal')
      axis.contour(X, Z, P, levels = [Pc], colors=color_sep,  labels='Separatrix')
      axis.add_patch(patches.Circle((0.,0.), 1, color=color_star, zorder=100, label="Star"))
      axis.axvline(parameters['R_lc'], color=color_lc, label='$R_{lc}$')
      axis.set_xlabel('x')
      axis.set_ylabel('z', rotation=0)
      axis.set_xlim ([0, 2*parameters['R_lc']])
      axis.set_ylim ([-parameters['R_lc'], parameters['R_lc']])
      axis.set_xticks(np.arange(0,31,5))
      axis.set_yticks(np.arange(-15,16,5))
      axis.set_title(titles[i])


   cbar_ticks_number = int(np.abs(max_value) + np.abs(min_value) + 1)
   cbar = fig.colorbar(cont1, ax=ax.ravel().tolist())
   cbar.set_ticks(np.logspace(min_value, max_value, cbar_ticks_number))

   return


# ===================================================================================================


def plot_scalar_functions(X, Z, P, Pc, T, T_prime, parameters):
    
   fig, axes = plt.subplots(2,2, sharex='col', sharey='row')

   # Exclude axis
   if parameters ['coordinates'] == 'spherical' or 'compactified':
      slc = np.s_[:,1:-1]
   else:
      slc = np.s_[:,:]

   # Set up
   line_levels = 20
   cmap_levels = 50
   color_star = '#CECFD2'
   color_lc = 'green'
   color_sep = 'red'
   color_lines = 'black'
   cmap = 'winter'
   titles = ["$P$", "$T$", "$T'$", "$TT'$"]
   scalars = [P, T, T_prime, T*T_prime]

   for i, ax in enumerate(axes.ravel()):
      
      ax.set_aspect ('equal')
      ax.contour(X, Z, P, levels = [Pc], colors=color_sep, labels='Separatrix')
      ax.add_patch(patches.Circle((0.,0.), 1, color=color_star, zorder=100, label="Star"))
      ax.axvline(parameters['R_lc'], color=color_lc, label='$R_{lc}$')
      ax.set_xlabel('x')
      ax.set_ylabel('z', rotation=0)
      ax.set_xlim ([0, 2*parameters['R_lc']])
      ax.set_ylim ([-parameters['R_lc'], parameters['R_lc']])
      ax.set_xticks(np.arange(0,31,5))
      ax.set_yticks(np.arange(-15,16,5))
      ax.set_title(titles[i])
      # cbar = fig.colorbar(cont[i], ax=axis)

      f = scalars[i][slc]
      # min_val, max_val = np.floor(np.log10(np.abs(f)[np.abs(f)>0].min())), np.ceil(np.log10(np.abs(f)[np.abs(f)>0].max()))
      min_val, max_val = -9, 3
      levels = np.logspace(min_val, max_val, cmap_levels)

      cont = ax.contour (X[slc], Z[slc]
                         , np.abs(f)
                         , levels=line_levels
                         , colors = color_lines
                        )

      
      contf = ax.contourf (X[slc], Z[slc]
                           , np.abs(f)
                           , levels=levels
                           , locator=ticker.LogLocator()
                           , cmap=cmap
                          )
      
      cbar = fig.colorbar(contf, ax=ax)
      cbar_ticks_number = int(np.abs(max_val) + np.abs(min_val) + 1)
      cbar.set_ticks(np.logspace(min_val, max_val, cbar_ticks_number))


   return


# ===================================================================================================


def plot_fields(X, Z, P, Pc, B_mag, E_mag, div_E, parameters):
    
    fig, axes = plt.subplots(1,2, sharex='col', sharey='row')

    # Exclude axis
    if parameters ['coordinates'] == 'spherical' or 'compactified':
        slc = np.s_[:,1:-1]
    else:
        slc = np.s_[:,:]

    # Set up
    cmap_levels = 50
    color_star = '#CECFD2'
    color_lc = 'green'
    color_sep = 'red'
    color_lines = 'black'
    cmap = 'coolwarm'
    titles = ['$B^2 - E^2$', '$\\nabla \cdot \\mathbf{E}$']
    scalars = [B_mag ** 2 - E_mag ** 2, div_E]
    linthresh = [1e-9, 1e-6]

    for i, ax in enumerate(axes.ravel()):
        
        ax.set_aspect ('equal')
        ax.contour(X, Z, P, levels=[Pc], colors=color_sep, labels='Separatrix')
        ax.add_patch(patches.Circle((0.,0.), 1, color=color_star, zorder=100, label="Star"))
        ax.axvline(parameters['R_lc'], color=color_lc, label='$R_{lc}$')
        ax.set_xlabel('x')
        ax.set_ylabel('z', rotation=0)
        ax.set_xlim ([0, 2*parameters['R_lc']])
        ax.set_ylim ([-parameters['R_lc'], parameters['R_lc']])
        ax.set_xticks(np.arange(0,31,5))
        ax.set_yticks(np.arange(-15,16,5))
        ax.set_title(titles[i])

        f = scalars[i][slc]
        contf = ax.contourf (X[slc], Z[slc]
                            , f
                            , cmap=cmap
                            , norm=colors.SymLogNorm(linthresh=linthresh[i], linscale=1)
                            , locator=ticker.SymmetricalLogLocator(linthresh=linthresh[i], base=10)
                            )
        
        cbar = fig.colorbar(contf, ax=ax, format=ticker.LogFormatterMathtext(), shrink=0.585)

    return