'''
This is for small utility functions that don't have a proper home yet
'''

import numpy as np

def radial_profile(data, center):
    '''
    Take the radial average of a 2D data array

    Taken from http://stackoverflow.com/a/21242776/5030014

    Parameters
    ----------
    data : ndarray (2D)
        the 2D array for which you want to calculate the radial average
    center : sequence
        the center about which you want to calculate the radial average

    Returns
    -------
    radialprofile : ndarray
        a 1D radial average of data
    '''

    y, x = np.indices((data.shape))

    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())

    nr = np.bincount(r.ravel())

    radialprofile = tbin / nr

    return radialprofile


def sliceMaker(y0, x0, width):
    '''
    A utility function to generate slices for later use.

    Parameters
    ----------
    y0 : int
        center y position of the slice
    x0 : int
        center x position of the slice
    width : int
        Width of the slice

    Returns
    -------
    slices : list
        A list of slice objects, the first one is for the y dimension and
        and the second is for the x dimension.

    Notes
    -----
    The method will automatically coerce slices into acceptable bounds.
    '''

    #calculate the start and end
    half1 = width//2
    #we need two halves for uneven widths
    half2 = width-half1
    ystart = y0 - half1
    xstart = x0 - half1
    yend = y0 + half2
    xend = x0 + half2

    toreturn = [slice(ystart,yend), slice(xstart, xend)]

    #return a list of slices
    return toreturn
