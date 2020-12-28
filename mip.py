# ! /usr/bin/env python
# -*- coding: utf-8 -*-
"""mip.py

A utility to call mip from the command line

Usage:
  mip.py <myfile> [--PDF --log]
  mip.py -h | --help

Options:
  -h --help     Show this screen.
  --PDF         Print PDF to current directory
  --log         Take log of data first

"""

from docopt import docopt
import matplotlib.pyplot as plt
from dphplotting.mip import mip

# check to see if we're being run from the command line
if __name__ == "__main__":
    # if we want to update this to take more arguments we'll need to use one of the
    # argument parsing packages

    # grab the arguments from the command line
    arg = docopt(__doc__)

    # a little output so that the user knows whats going on
    print("Running mip on", arg["<myfile>"])

    # Need to take the first system argument as the filename for a TIF file

    # test if filename has tiff in it
    filename = arg["<myfile>"]

    # try our main block
    try:
        if ".tif" in filename or ".tiff" in filename:
            # Import skimage so we have access to tiff loading
            import tifffile as tif

            # here's the real danger zone, did the user give us a real file?
            try:
                data = tif.imread(filename)
            except FileNotFoundError as er:
                raise er
            if arg["--log"]:
                import numpy as np

                if data.min() > 0:
                    data = np.log(data)
                else:
                    print(filename, "had negative numbers, log not taken")

            # Trying to set the cmap here opens a new figure window
            # need to set up kwargs for efficient argument passing
            # plt.set_cmap('gnuplot2')
            # plot the data
            fig, ax = mip(data)
            # readjust the white space (maybe move this into main code later)
            fig.subplots_adjust(top=0.85, hspace=0.3, wspace=0.3)
            # add an overall figure title that's the file name
            fig.suptitle(filename, fontsize=16)

            # check to see if we should make a PDF
            if arg["--PDF"]:
                fig.savefig(filename.replace(".tiff", ".pdf").replace(".tif", ".pdf"))
            else:
                # I still don't know why fig.show() doesn't work
                # I think it has something to do with pyplot's backend
                plt.show()
        else:
            # this is our own baby error handling, it avoids loading the
            # skimage package
            print("You didn't give me a TIFF")

    except Exception as er:
        print(er)
