import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

def plot_2D_with_slices(X,Y,Z,range=1,
                        x_axis={"title": "x_title","scale": "linear", "slice": 5},
                        y_axis={"title": "y_title","scale": "linear", "slice": 5}):
    
    # Create the figure and subplots
    cm = 1/2.54  # centimeters in inches
    fig, axs = plt.subplots(2, 2, figsize=(12*cm, 12*cm), gridspec_kw={'height_ratios': [1, 2], 'width_ratios': [2, 1]})

    pcm = axs[1, 0].pcolormesh(X, Y, Z, cmap='seismic',norm = Normalize(vmin=-range, vmax=range))
    cbar_ax = fig.add_axes([0.65, 0.8, 0.3, 0.05])  # Position of the colorbar
    fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal', label='Polarization')

    index_X = (np.abs(X - x_axis["slice"])).argmin()
    index_Y = (np.abs(Y - y_axis["slice"])).argmin()

    axs[1, 0].axhline(y=Y[index_Y],c="c",linewidth=2)
    axs[1, 0].axvline(x=X[index_X],c="m",linewidth=2)
    axs[1, 0].set_xlabel(x_axis["title"],fontsize=16)
    axs[1, 0].set_ylabel(y_axis["title"],fontsize=16)

    axs[1, 0].set_yscale(y_axis["scale"])
    axs[1, 0].set_xscale(x_axis["scale"])
    # Add ticks on the right and top axes of the 2D graph
    axs[1, 0].tick_params(axis='x', direction='out', length=5, width=1, top=True)
    axs[1, 0].tick_params(axis='y', direction='out', length=5, width=1, right=True)

    # Plot the horizontal slice
    axs[0, 0].plot(X, Z[index_Y, :], color='c')
    #axs[0, 0].set_title('Horizontal Slice')
    axs[0, 0].set_xlim([min(X), max(X)])  # Adjust x-axis limits if necessary
    axs[0, 0].set_ylim([-range, range])  # Adjust y-axis limits if necessary
    #axs[0, 0].set_xscale(x_axis["scale"])

    # Plot the vertical slice
    axs[1, 1].plot(Z[:, index_X], Y, color='m')
    #axs[1, 1].set_title('Vertical Slice')
    axs[1, 1].set_ylim([min(Y), max(Y)])  # Adjust y-axis limits if necessary
    axs[1, 1].set_xlim([-range, range])  # Adjust x-axis limits if necessary
    axs[1, 1].set_yscale(y_axis["scale"])

    # Remove the empty subplot in the top right corner
    fig.delaxes(axs[0, 1])

    plt.tight_layout()
    #plt.savefig('example_plot.tiff', dpi=300, format='tiff')
    plt.show()

def plot_2D(X,Y,Z,range=1,
                        x_axis={"title": "x_title","scale": "linear"},
                        y_axis={"title": "y_title","scale": "linear"}):
    
    # Create the figure and subplots
    cm = 1/2.54  # centimeters in inches
    fig, axs = plt.subplots(1, 1, figsize=(12*cm, 12*cm))
    
    pcm = axs.pcolormesh(X, Y, Z, cmap='seismic',norm = Normalize(vmin=-range, vmax=range))
    cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.5])  # Position of the colorbar
    fig.colorbar(pcm, cax=cbar_ax, orientation='vertical', label='Polarization')


    axs.set_xlabel(x_axis["title"],fontsize=18)
    axs.set_ylabel(y_axis["title"],fontsize=18)

    axs.set_yscale(y_axis["scale"])
    axs.set_xscale(x_axis["scale"])

    #plt.tight_layout()
    #plt.savefig('example_plot.tiff', dpi=300, format='tiff')
    plt.subplots_adjust(top = 0.80, right = 0.80)
    plt.show()