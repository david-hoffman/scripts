# ! /usr/bin/env python
#  -*- coding: utf-8 -*-
"""simcheck.py

A utility to call simcheck from the command line

Usage:
    simcheck.py <myfile> -n <numphases> [--png --raw]
    simcheck.py -h | --help

Options:
    -h --help     Show this screen.
    -n            The number of phases in the SIM stack
    --png         save PNGs

"""

from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np
import os
import Mrc
from dphplotting import display_grid
from scipy.ndimage import gaussian_filter
from pyfftw.interfaces.scipy_fftpack import ifftshift, fftshift, fftn
from matplotlib.colors import LogNorm

# check to see if we're being run from the command line
if __name__ == '__main__':

    # if we want to update this to take more arguments we'll need to use one of
    # the argument parsing packages
    # grab the arguments from the command line
    arg = docopt(__doc__)

    # a little output so that the user knows whats going on
    print('Running simcheck on', arg['<myfile>'])

    # Need to take the first system argument as the filename for a TIF file

    # test if filename has mrc in it
    filename = arg['<myfile>']

    name = os.path.basename(os.path.split(os.path.abspath(filename))[0])

    data = Mrc.Mrc(filename).data

    nphases = int(arg['<numphases>'])
    norients = data.shape[0]//nphases       # need to use integer divide

    # check user input before proceeding
    assert nphases*norients == data.shape[0]

    newshape = norients, nphases, data.shape[-2], data.shape[-1]

    # FT data only along spatial dimensions
    ft_data = ifftshift(
                fftn(fftshift(
                    data, axes=(1, 2)),
                 axes=(1, 2)),
            axes=(1, 2))
    # average only along phase, **not** orientation
    # This should be the equivalent of the FT of the average image per each
    # phase (i.e it should be symmetric as the phases will have averaged out)
    ft_data_avg = ft_data.reshape(newshape).mean(1)
    # Do the same, but take the absolute value before averaging, in this case
    # the signal should add up because the phase has been removed
    ft_data_avg_abs = np.abs(ft_data).reshape(newshape).mean(1)
    # Take the difference of the average power and the power of the average
    ft_data_diff = ft_data_avg_abs-abs(ft_data_avg)

    orients = ['Orientation {}'.format(i) for i in range(norients)]
    # Plot everything and save
    fig1, ax1 = display_grid({k: v/v.max()
                              for k, v in zip(orients, ft_data_avg_abs)},
                             figsize=6, cmap='gnuplot2', norm=LogNorm())
    filename1 = name + ' Average of Powers'
    fig1.suptitle(filename1, fontsize=16)

    fig2, ax2 = display_grid({k: abs(v/v.max())
                              for k, v in zip(orients, ft_data_avg)},
                             figsize=6, cmap='gnuplot2', norm=LogNorm())
    filename2 = name + ' Power of Average'
    fig2.suptitle(filename2, fontsize=16)

    #  fig3, ax3 = display_grid({k : v for k, v in zip(orients, ft_data_diff)}, figsize=6, 
    #          cmap ='bwr', vmin = ft_data_diff.min(), vmax = ft_data_diff.max())
    #  filename3 = name + ' Difference of Averages'
    #  fig3.suptitle(filename3, fontsize=16)

    #  fig4, ax4 = display_grid({k : median_filter(v, 3) for k, v in zip(orients, ft_data_diff)}, figsize=6, 
    #               cmap ='bwr')
    #  filename4 = name + ' Difference of Averages - Median Filter'
    #  fig4.suptitle(filename4, fontsize=16)

    def scale(v):
        mymin = v.min()
        mymax = v.max()
        return (v-mymin)/(mymax - mymin)

    filtered_ft_data_diff = gaussian_filter(ft_data_diff, (0, 6, 6))
    fig5, ax5 = display_grid({k: scale(v)
                              for k, v in zip(orients, filtered_ft_data_diff)},
                             figsize=6, cmap='bwr')
    filename5 = name + ' Difference of Averages - Gaussian Filter'
    fig5.suptitle(filename5, fontsize=16)

    # Plot and save the raw data
    if arg['--raw']:
        fig, ax = display_grid({k: v for k, v in zip(
                ['Phase {}, Orientation {}'.format(i, j)
                 for i in range(nphases) for j in range(norients)], data)},
                  figsize=3, cmap='gnuplot2', vmin=data.min(), vmax=data.max())
        filename6 = name + ' Raw Data'
        fig.suptitle(filename6, fontsize=16)

    if arg['--png']:
        fig1.savefig(filename1+'.png', dpi=300, transparent=True,
                     bbox_inches='tight')
        fig2.savefig(filename2+'.png', dpi=300,
                     transparent=True, bbox_inches='tight')
        #  fig3.savefig(filename3+'.png', dpi = 300, transparent = True, bbox_inches = 'tight')
        #  fig4.savefig(filename4+'.png', dpi = 300, transparent = True, bbox_inches = 'tight')
        fig5.savefig(filename5+'.png', dpi=300,
                     transparent=True, bbox_inches='tight')
        if arg['--raw']:
            fig.savefig(filename6+'.png', dpi=300,
                        transparent=True, bbox_inches='tight')
    else:
        plt.show()
