#!/usr/bin/env python
# -*- coding: utf-8 -*-
# decay_utils.py
"""
Some utility functions for analyzing decay data

Copyright (c) 2017, David Hoffman
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.colors import ListedColormap, LogNorm
from simrecon_utils import split_img, combine_img
from dphutils import scale
from skimage.exposure import adjust_gamma

# make a new colormap that is like "Greys_r" but with an alpha chanel equal to
# the greylevel
greys_alpha_cm = ListedColormap([(i / 255,) * 3 + ((255 - i) / 255,)
                                 for i in range(256)])


def exp(x, A, k, O):
    """Exponential utility function for curve_fit"""
    return O + A * np.exp(-x / k)


def recovery_plot(k, v, bg=100, p0=(-.1, 100, 1), num_tiles=16):
    """Plot the recovery of intensity

    Parameters
    ----------
    k : str
        Figure title
    v : nd.array (t, y, x)
        Assumes an image stack
    bg : numeric
        Background for image data
    p0 : tuple
        Initial guesses for exponential decay
    num_tiles : int
        The number of sub images to calculate

    Returns
    -------
    fig : figure handle
    axs : axes handles"""
    # make figure
    fig, axs = plt.subplots(1, 4, figsize=(4 * 3.3, 3),
                            gridspec_kw=dict(width_ratios=(1, 1.3, 1, 1)))
    (ax, ax_k, ax_h, ax_i) = axs
    my_shape = v.shape
    fig.suptitle(k, y=1.05)
    # split the image
    img_split = split_img(v, v.shape[-1] // num_tiles) * 1.0 - bg
    # sum kinetics, convert to float
    kinetics = img_split.mean((2, 3))
    norm_kinetics = kinetics / kinetics[:, np.newaxis].max(-1)
    xdata = np.arange(my_shape[0]) * 0.1 * 46
    ks = np.array([curve_fit(exp, xdata, kin, p0=p0)[0][1]
                   for kin in norm_kinetics])
    kin_img = np.ones_like(img_split)[:, 0, ...] * ks[:, np.newaxis, np.newaxis]
    # plot kinetics, color by amount of bleaching and set alpha to initial intensity
    for trace, cpoint in zip(norm_kinetics, scale(ks)):
        if np.isfinite(cpoint):
            ax.plot(xdata, trace, c=plt.get_cmap("spring")(cpoint))
    # start plotting
    ax.set_title("Bleaching Kinetics")
    ax.set_xlabel("J/cm$^2$")
    ax.set_ylabel("Relative Intensity (a.u.)")
    ax.tick_params()
    img = ax_k.matshow(combine_img(kin_img), cmap="spring")
    cb = plt.colorbar(img, ax=ax_k, aspect=10, use_gridspec=True)
    cb.set_label("Decay constant (J/cm$^2$)")
    ax_k.matshow(v.max(0), cmap=greys_alpha_cm, norm=LogNorm())
    ax_i.matshow(adjust_gamma(v.max(0), 0.25), cmap="Greys_r")
    ax_h.hist(ks, bins=int(np.sqrt(ks.size)))
    ax_h.set_title("Median = {:.0f}".format(np.median(ks)))
    for ax in (ax_i, ax_k):
        ax.grid("off")
        ax.axis("off")
    ax_i.set_title("Image")
    ax_k.set_title("Bleaching Map")
    fig.tight_layout()
    return fig, axs


def bleach_plot(k, v, bg=100.0):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3),
                            gridspec_kw=dict(width_ratios=(1, 1.25)))
    ax, ax_k = axs
    fig.suptitle(k, y=1)
    # calculate and norm kinetics on a per pixel basis
    kinetics = (v * 1.0 - bg)
    kinetics[kinetics < 0] = np.nan
    kinetics /= np.nanmax(kinetics)
    # show max projection
    ax.matshow(adjust_gamma(v.max(0), 0.25), cmap="Greys_r")
    # show kinetics, inside (0,1)
    img = ax_k.matshow(kinetics[-1], vmin=0, vmax=1, cmap="spring")
    ax_k.matshow(v.max(0), cmap=greys_alpha_cm, norm=LogNorm())
    ax.set_title("Image")
    ax_k.set_title("Kinetics")
    plt.colorbar(img, ax=ax_k, aspect=10, shrink=0.75, use_gridspec=True)
    for ax in axs:
        ax.axis("off")
    fig.tight_layout()
    return fig, axs


def bleach_plot2(k, v, bg=100.0, num_tiles=16, gamma=0.5, dt=1):
    """Plot bleaching of image timeseries

    Parameters
    ----------
    k : str
        Title string
    v : ndarray (t, y, x)
        Image stack (x must y)
    bg : float
        Background counts
    num_tiles : int
        Number of tiles per image side
    """
    nt, ny, nx = v.shape
    assert ny == nx, "data isn't square"
    # make figure
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    (ax, ax_k, ax_i) = axs
    fig.suptitle(k, y=1.05)
    # split the image
    img_split = split_img(v, nx // num_tiles)
    # sum kinetics, convert to float
    bg = np.asarray(bg)
    bg.shape = (-1, 1, 1)
    kinetics = (img_split * 1.0 - bg).sum((2, 3))
    # anything that's negative mask with an nan
    kinetics[kinetics < 0] = np.nan
    # normalize all the kinetics with the max one
    norm_kinetics = kinetics / np.max(kinetics[:, np.newaxis], -1)
    kin_img = np.ones_like(img_split)[:, 0, ...] * norm_kinetics[:, -1, np.newaxis, np.newaxis]
    ### start plotting
    # plot kinetics, color by amount of bleaching and set alpha to initial intensity
    for trace, cpoint in zip(norm_kinetics, scale(norm_kinetics[:, -1])):
        # make sure the point is not an nan
        if np.isfinite(cpoint):
            ax.loglog(np.arange(len(trace)) * dt, trace, c=plt.get_cmap("spring")(cpoint))
    # label stuff
    ax.set_title("Bleaching Kinetics")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative Intensity (a.u.)")
    ax.tick_params()
    # calculate the max image
    v_max = v.max(0)
    # plot color coded kinetics
    ax_k.matshow(combine_img(kin_img), cmap="spring", zorder=0)
    # plot image with alpha over it
    ax_k.matshow(v_max, cmap=greys_alpha_cm, norm=LogNorm(), zorder=1)
    # plot raw image with gamma adjustment for reference
    ax_i.matshow(adjust_gamma(v_max, gamma), cmap="Greys_r")
    # remove grids
    for ax in (ax_i, ax_k):
        ax.grid("off")
        ax.axis("off")
    # set titles
    ax_i.set_title("Image")
    ax_k.set_title("Bleaching Map")
    fig.tight_layout()
    return fig, axs

def gen_wavelength(center_wl, start_p=0, end_p=2048, center_p=1024, pix_size=0.0065, grating_pitch=300.0):
    """Generate a wavelength axis for spectroscopic data

    Parameters
    ----------
    center_wl : float
        Center wavelength as recorded by LabVIEW
    start_p : int
        first pixel
    end_p : int
        last pixel
    center_p : int
        center pixel
    pix_size : float
        pixel size in microns
    grating_pitch : float
        grooves per mm of the grating
    offset : float

    Returns
    -------
    wl : ndarray
        Wavelength axis in nm
    """
    dispersion_1200 = 1.6
    nm_per_mm = dispersion_1200 * 1200 / grating_pitch
    nm_per_pixel = nm_per_mm * pix_size
    pix = np.arange(start_p, end_p) - center_p
    return pix * nm_per_pixel + center_wl
