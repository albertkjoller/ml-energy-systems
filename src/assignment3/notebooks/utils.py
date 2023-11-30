import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def add_colorbar(ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

def add_binary_colorbar(ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="20%", pad=0.2)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['0', '1'])

def add_error_colorbar(ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="30%", pad=0.2)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['-1', '0', '1'])
