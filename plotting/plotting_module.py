
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

   custom_colors = ['#386cb0', '#f0027f', '#bf5b17', '#666666', '#7fc97f', '#beaed4', '#fdc086', '#ffff99']
   plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)

   plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
   plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
   plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
   plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
   plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
   plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
   plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title

   return


# ===================================================================================================


def plot_elements(ax, X, Z, P, Pc, parameters, surface=True, separatrix=True, lc=True):
   """
      Plot the various elements of the pulsar magnetosphere.
   """
   
   # Choose colors
   color_star = 'lightsteelblue'
   color_lc = 'gold'
   color_sep = 'firebrick'

   # Plot surface
   if surface:
      ax.add_patch(patches.Circle((0.,0.), 1, color=color_star, zorder=100, label="Star"))
   
   # Plot the separatrix
   if separatrix:
      ax.contour(X, Z, P, levels=[Pc], linewidths=3, colors=color_sep, zorder=99, labels='Separatrix')
   
   # Plot the light cylinder
   if lc:
      ax.axvline(parameters['R_lc'], color=color_lc, zorder=98, label='$R_{lc}$')

   return


# ===================================================================================================


def plot_pinn(X, Z, P, Pc, fb, hb, Np, parameters):
   
   '''
      Plot the three functions that define the parametrisation (fb, hb and Np).
   '''
   
   fig, axes = plt.subplots(2,2, sharex='col', sharey='row')

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
      
      # Plot pulsar magnetosphere elements
      plot_elements(ax, X, Z, P, Pc, parameters)

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


def plot_scalar_functions(X, Z, P, Pc, T, Tprime, parameters):
    
   fig, axes = plt.subplots(2,2, sharex='col', sharey='row')
   
   # Properties of contour lines
   lines_levels = 10
   lines_color = 'black'

   # Properties of colormap
   cmap_levels = 100
   cmap = 'twilight'
   
   # Properties of each individual figure
   titles = ["$P$", "$T$", "$T'$", "$TT'$"]
   scalars = [P, T, Tprime, T*Tprime]

   # Loop over all quantities to be plotted
   for i, ax in enumerate(axes.ravel()):
      
      # Plot pulsar magnetosphere elements
      plot_elements(ax, X, Z, P, Pc, parameters)

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
      mask = np.isfinite(np.abs(f))
      f = np.where(mask, f, 0)

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


def plot_fields(X, Z, P, Pc, B_mag, E_mag, div_E, div_B, parameters):
    
   fig, axes = plt.subplots(1,2, sharex='col', sharey='row')

   # Set up
   slc = np.s_[:,1:-1]


   cmap_levels = 50

   color_lines = 'black'
   cmap = 'coolwarm_r'
   # titles = ['$B^2 - E^2$', '$\\nabla \cdot \\mathbf{E}$']
   titles = ['$\\nabla \cdot \\mathbf{B}$', '$\\nabla \cdot \\mathbf{E}$']

   # scalars = [B_mag ** 2 - E_mag ** 2, div_E]
   scalars = [div_B, div_E]

   for i, ax in enumerate(axes.ravel()):
      
      # Plot pulsar magnetosphere elements
      plot_elements(ax, X, Z, P, Pc, parameters)


      ax.set_xlabel('x')
      ax.set_ylabel('z', rotation=0)
      ax.set_xlim ([0, 2*parameters['R_lc']])
      ax.set_ylim ([-parameters['R_lc'], parameters['R_lc']])
      ax.set_xticks(np.arange(0,31,5))
      ax.set_yticks(np.arange(-15,16,5))
      ax.set_title(titles[i])

      f = (scalars[i][slc])
      mask = np.isfinite(f)
      min_val = np.floor(np.log10(np.abs(scalars[1][slc])[mask].min()))
      max_val = np.ceil(np.log10(np.abs(scalars[1][slc])[mask].max()))
      min_val = -7
      max_val = 0
      cmap_levels = np.logspace(min_val, max_val, 50)
      cmap_levels = np.hstack([-cmap_levels[::-1], cmap_levels])
                           
      contf = ax.contourf (X[slc], Z[slc]
                           , f
                           , levels=cmap_levels
                           , norm=colors.SymLogNorm(linthresh=10**min_val, linscale=1)
                           , locator=ticker.SymmetricalLogLocator(linthresh=10**min_val, base=10)
                           , cmap=cmap
                           )
      
      cbar = fig.colorbar(contf, ax=ax, format=ticker.LogFormatterMathtext())
      cbar_ticks_number = int(np.abs(max_val - min_val + 1))
      ticks = np.logspace(min_val, max_val, cbar_ticks_number)
      cbar.set_ticks(np.hstack([-ticks[-1:0:-3], 0, ticks[2::3]]))
   return


# ===================================================================================================


def plot_pulsar_equation (X, Z, pulsar_eq, P, Pc, parameters):
    
   fig, ax = plt.subplots()

   # Set up
   slc = np.s_[:,1:-1]
   mask = np.isfinite(pulsar_eq[slc] / P[slc])
   # min_val = np.floor(np.log10(np.abs(pulsar_eq[slc][mask]).min()))
   min_val = -6
   max_val = np.ceil(np.log10(np.abs(pulsar_eq[slc][mask]).max()))
   levels = np.logspace(min_val, max_val, 50)
   
   
   color_lines = 'black'
   cmap = 'copper'

   # Plot pulsar magnetosphere elements
   plot_elements(ax, X, Z, P, Pc, parameters)

   ax.set_xlabel('x')
   ax.set_ylabel('z', rotation=0)
   ax.set_xlim ([0, 2*parameters['R_lc']])
   ax.set_ylim ([-parameters['R_lc'], parameters['R_lc']])
   ax.set_xticks(np.arange(0,31,5))
   ax.set_yticks(np.arange(-15,16,5))

   contf = ax.contourf(X[slc], Z[slc]
                       , np.abs(pulsar_eq[slc])
                       , cmap=cmap
                       , norm=colors.LogNorm()
                       , levels=levels
                       , extend='min'
                      )

   cbar = fig.colorbar(contf)
   cbar_ticks_number = int(np.abs(max_val - min_val + 1))
   cbar.set_ticks(np.logspace(min_val, max_val, cbar_ticks_number))

   return


# ===================================================================================================


def plot_T_of_P (P, Pc, T, Tprime, parameters):
    
   fig, axes = plt.subplots(1, 2)

   #Calculate monopole solution for normalisation and comparison
   # P_mon = 1 / parameters['R_lc'] (1 - )
   scalars = [-T, T * Tprime]
   
   for i, ax in enumerate(axes):
      f = scalars[i].ravel()

      ax.set_xlim([0, 1.25 * Pc])
      ax.axvline(Pc, c='k', alpha=0.5, ls='--')

      ax.scatter(P.ravel(), f, s=10, marker='*', color='#1b9e77')
   

   return


