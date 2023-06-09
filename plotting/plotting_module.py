
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
   plt.rcParams["text.latex.preamble"] = [r'\usepackage{xfrac}', r'\usepackage{bm}']

   plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
   plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
   plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
   plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
   plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
   plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
   plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title

   return


# ===================================================================================================



def plot_pinn(X, Z, P, Pc, fb, hb, Np, parameters):
   
   '''
      Plot the three functions that define the parametrisation (fb, hb and Np).
   '''
   
   fig, axes = plt.subplots(2,2, sharex='col', sharey='row')
   
   # Properties of surface, separatrix and LC
   color_star = '#CECFD2'
   color_lc = 'green'
   color_sep = 'red'

   # Properties of contour lines
   lines_levels = 10
   lines_color = 'black'

   # Properties of colormap
   cmap_levels = 50
   cmap = 'ocean'
   
   # Properties of each individual figure
   titles = ["$f_b$", "$h_b$", "$N_P$", "$\\frac{h_b N_P}{F_b}$"]
   scalars = [fb, hb, Np, hb * Np / fb]

   # Loop over all quantities to be plotted
   for i, ax in enumerate(axes.ravel()):
      
      # Set equal aspect ratio in all figures 
      # ax.set_aspect ('equal')

      # Plot separatrix
      ax.contour(X, Z, P, levels = [Pc], colors=color_sep, labels='Separatrix')

      # Plot surface
      ax.add_patch(patches.Circle((0.,0.), 1, color=color_star, zorder=100, label="Star"))

      # Plot LC
      ax.axvline(parameters['R_lc'], color=color_lc, label='$R_{lc}$')

      # Set plot limits
      ax.set_xlim ([0, 2*parameters['R_lc']])
      ax.set_ylim ([-parameters['R_lc'], parameters['R_lc']])

      # Label axis (only in the borders of the plots)
      if i == 2 or i == 3:
         ax.set_xlabel('x')
      if i == 0 or i == 2:
         ax.set_ylabel('z', rotation=0)
      
      # Set axis ticks
      ax.set_xticks(np.arange(0,31,5))
      ax.set_yticks(np.arange(-15,16,5))

      # Set figure titles
      ax.set_title(titles[i])

      # Exclude axis, infs and nans
      slc = np.s_[:,1:-1]
      # f = np.abs(scalars[i][slc])
      f = scalars[i][slc]
      # mask = np.isfinite(np.abs(f))
      # f = np.where(mask, f, np.nan)

      # Choose which field lines will be plotted (plot only in open region)
      # f_lines = np.where(P[slc] <= Pc, f, np.nan)
      f_lines = f

      # Get minimum and maximum value to be shown in colormap
      min_val = np.floor(np.log10(f.min()))
      min_val = -2
      max_val = np.ceil(np.log10(f.max()))

      # Set the colormap levels
      cmap_levels = np.logspace(min_val, max_val, 50)
      cmap_levels = np.hstack([-cmap_levels[::-1], cmap_levels])

      # Plot field lines
      cont = ax.contour (X[slc], Z[slc]
                         , f_lines
                         , levels=lines_levels
                         , colors=lines_color
                         , alpha=0.5
                        )

      # Plot colormap
      contf = ax.contourf (X[slc], Z[slc]
                           , f
                           , levels=cmap_levels
                           , norm=colors.SymLogNorm(linthresh=10**min_val, linscale=1)
                           , locator=ticker.SymmetricalLogLocator(linthresh=10**min_val, base=10)
                           , cmap=cmap
                          )
      
      # Set the colorbar
      cbar = fig.colorbar(contf, ax=ax, format=ticker.LogFormatterMathtext())
      cbar_ticks_number = int(np.abs(max_val - min_val + 1))
      ticks = np.logspace(min_val, max_val, cbar_ticks_number)
      cbar.set_ticks(np.hstack([-ticks[-1:0:-1], 0, ticks[1:]]))


   return


# ===================================================================================================


def plot_scalar_functions(X, Z, P, Pc, T, T_prime, parameters):
    
   fig, axes = plt.subplots(2,2, sharex='col', sharey='row')
   
   # Properties of surface, separatrix and LC
   color_star = '#CECFD2'
   color_lc = 'green'
   color_sep = 'red'

   # Properties of contour lines
   lines_levels = 10
   lines_color = 'black'

   # Properties of colormap
   cmap_levels = 100
   cmap = 'twilight'
   
   # Properties of each individual figure
   titles = ["$P$", "$T$", "$T'$", "$TT'$"]
   scalars = [P, T, T_prime, T*T_prime]

   # Loop over all quantities to be plotted
   for i, ax in enumerate(axes.ravel()):
      
      # Set equal aspect ratio in all figures 
      # ax.set_aspect ('equal')

      # Plot separatrix
      ax.contour(X, Z, P, levels = [Pc], colors=color_sep, labels='Separatrix')

      # Plot surface
      ax.add_patch(patches.Circle((0.,0.), 1, color=color_star, zorder=100, label="Star"))

      # Plot LC
      ax.axvline(parameters['R_lc'], color=color_lc, label='$R_{lc}$')

      # Set plot limits
      ax.set_xlim ([0, 2*parameters['R_lc']])
      ax.set_ylim ([-parameters['R_lc'], parameters['R_lc']])

      # Label axis (only in the borders of the plots)
      if i == 2 or i == 3:
         ax.set_xlabel('x')
      if i == 0 or i == 2:
         ax.set_ylabel('z', rotation=0)
      
      # Set axis ticks
      ax.set_xticks(np.arange(0,31,5))
      ax.set_yticks(np.arange(-15,16,5))

      # Set figure titles
      ax.set_title(titles[i])

      # Exclude axis, infs and nans
      slc = np.s_[:,1:-1]
      f = np.abs(scalars[i][slc])
      # mask = np.isfinite(np.abs(f))
      # f = np.where(mask, f, np.nan)

      # Choose which field lines will be plotted (plot only in open region)
      f_lines = np.where(P[slc] <= Pc, f, np.nan)

      # Get minimum and maximum value to be shown in colormap
      # min_val = np.floor(np.log10(np.abs(f[mask]).min()))
      min_val = -7
      max_val = np.ceil(np.log10(f.max()))

      # Set the colormap levels
      cmap_levels = np.logspace(min_val, max_val, 50)

      # Plot field lines
      cont = ax.contour (X[slc], Z[slc]
                         , f_lines
                         , levels=lines_levels
                         , colors=lines_color
                         , alpha=0.5
                        )

      # Plot colormap
      contf = ax.contourf (X[slc], Z[slc]
                           , f
                           , levels=cmap_levels
                           , locator=ticker.LogLocator()
                           , cmap=cmap
                          )
      
      # Set the colorbar
      cbar = fig.colorbar(contf, ax=ax)
      cbar_ticks_number = int(np.abs(max_val - min_val + 1))
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
        
      # ax.set_aspect ('equal')
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


# ===================================================================================================


def plot_pulsar_equation (X, Z, pulsar_eq, P, Pc, parameters):
    
   fig, ax = plt.subplots()

   # Set up
   slc = np.s_[1:-1,1:-1]
   mask = np.isfinite(pulsar_eq[slc] / P[slc])
   # min_val = np.floor(np.log10(np.abs(pulsar_eq[slc][mask]).min()))
   min_val = -6
   max_val = np.ceil(np.log10(np.abs(pulsar_eq[slc][mask] / P[slc][mask]).max()))-1
   levels = np.logspace(min_val, max_val, 50)
   color_star = '#CECFD2'
   color_lc = 'green'
   color_sep = 'red'
   color_lines = 'black'
   cmap = 'copper'
        
   # ax.set_aspect ('equal')
   # ax.contour(X, Z, P, levels=[Pc], colors=color_sep, labels='Separatrix')
   ax.add_patch(patches.Circle((0.,0.), 1, color=color_star, zorder=100, label="Star"))
   # ax.axvline(parameters['R_lc'], color=color_lc, label='$R_{lc}$')
   ax.set_xlabel('x')
   ax.set_ylabel('z', rotation=0)
   ax.set_xlim ([0, 2*parameters['R_lc']])
   ax.set_ylim ([-parameters['R_lc'], parameters['R_lc']])
   ax.set_xticks(np.arange(0,31,5))
   ax.set_yticks(np.arange(-15,16,5))

   contf = ax.contourf(X[slc], Z[slc]
                       , np.abs(pulsar_eq[slc] / P[slc])
                       , cmap=cmap
                       , norm=colors.LogNorm(vmin=10**min_val, vmax=10**max_val)
                       , levels=levels
                       , extend='both'
                      )

   cbar = fig.colorbar(contf)
   cbar_ticks_number = int(np.abs(max_val - min_val + 1))
   cbar.set_ticks(np.logspace(min_val, max_val, cbar_ticks_number))

   return


# ===================================================================================================


def plot_T_of_P (P, Pc, T, T_prime, parameters):
    
   fig, ax = plt.subplots(1, 1)

   # Mask T_prime to hide divergent values due to numerical derivation
   mask = np.abs(T_prime) < 0.15
   T_prime_masked = np.where(mask, T_prime, np.nan)
   
   #Calculate monopole solution for normalisation and comparison
   # P_mon = 1 / parameters['R_lc'] (1 - )
   scalars = [-T, -T_prime, T * T_prime]
   scalars = [-T]
   # scalars_monopole = [-T, -T_prime_masked, T * T_prime_masked]

   # for i, ax in enumerate(axes):
      # f = scalars[i].ravel()

   # ax.set_xlim([0, 1.1 * Pc])
   ax.axvline(Pc, c='k', alpha=0.5, ls='--')

   ax.scatter(P.ravel(), -T, s=10, marker='*')
   fig, ax = plt.subplots(1, 1)

   ax.axvline(Pc, c='k', alpha=0.5, ls='--')

   ax.scatter(P.ravel(), -T*T_prime, s=10, marker='*')

   return


