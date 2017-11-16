# # PALM Blinking and Decay Analysis
# The purpose of this notebook is to analyze PALM diagnostic data in a consistent way across data sets. 
# The basic experiment being analyzed here is data that has been reactivated and deactivated multiple times.

import gc
import json
import os
import numpy as np
import pandas as pd
# regular plotting
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.colors import LogNorm, PowerNorm, ListedColormap
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# data loading
from scipy.io import readsav
from skimage.external import tifffile as tif

# get multiprocessing support
import dask
import dask.array
from dask.diagnostics import ProgressBar
import dask.multiprocessing

# need to be able to remove fiducials
from peaks.peakfinder import PeakFinder
import tqdm

# Need partial
from functools import partial

# need otsu
from skimage.filters import threshold_otsu, threshold_triangle

# need ndimage
import scipy.ndimage as ndi

from scipy.misc import imsave
from dphutils import mode, _calc_pad, scale
from dphplotting import auto_adjust
from scipy.optimize import curve_fit
from pyPALM.drift import remove_xy_mean, calc_drift, calc_fiducial_stats, extract_fiducials, plot_stats

# for fast hist
from numpy.core import atleast_1d, atleast_2d
from numba import njit
from scipy.spatial import cKDTree


greys_alpha_cm = ListedColormap([(i / 255,) * 3 + ((255 - i) / 255,) for i in range(256)])


def peakselector_df(path, verbose=False):
    """Read a peakselector file into a pandas dataframe"""
    if verbose:
        print("Reading {} into memory ... ".format(path))
    sav = readsav(path, verbose=verbose)
    # pull out cgroupparams, set the byteorder to native and set the rownames
    df = pd.DataFrame(sav["cgroupparams"].byteswap().newbyteorder(), columns=sav["rownames"].astype(str))
    return df

pb = ProgressBar()


def print_maxmin(k, df):
    print("{:=^60}".format("> {} <".format(k)))
    print("{:-^60}".format("> Max <"))
    print(df.max())
    print("{:-^60}".format("> Min <"))
    print(df.min())


def grouped_peaks(df):
    """Return a DataFrame with only grouped peaks."""
    return df[df["Frame Index in Grp"] == 1]


def calc_bins(raw_data, subsampling=10):
    """Calculate the right bins from the raw data given a subsampling factor"""
    nz, ny, nx = raw_data.shape
    sub_sample_xy = 1 / subsampling
    xbins = np.arange(0, nx + sub_sample_xy, sub_sample_xy)
    ybins = np.arange(0, ny + sub_sample_xy, sub_sample_xy)
    return ybins, xbins


def filter_fiducials(df, blobs, radius):
    """Do the actual filtering
    
    We're doing it sequentially because we may run out of memory.
    If initial DataFrame is 18 GB (1 GB per column) and we have 200 """
    blob_filt = pd.Series(np.ones(len(df)), index=df.index, dtype=bool)
    for i, (y, x) in enumerate(tqdm.tqdm_notebook(blobs, leave=False, desc="Filtering Fiducials")):
        bead_filter = np.sqrt((df.x0 - x) ** 2 + (df.y0 - y) ** 2) > radius
        blob_filt &= bead_filter
        del bead_filter
        if not i % 10:
            gc.collect()
    gc.collect()
    df = df[blob_filt]
    return df


def extract_fiducials(df, blobs, radius, min_num_frames=0):
    """Do the actual filtering
    
    We're doing it sequentially because we may run out of memory.
    If initial DataFrame is 18 GB (1 GB per column) and we have 200 """
    fiducials_dfs = [df[np.sqrt((df.x0 - x) ** 2 + (df.y0 - y) ** 2) < radius]
        for y, x in tqdm.tqdm_notebook(blobs, leave=True, desc="Extracting Fiducials")]
#     # remove any duplicates in a given frame by only keeping the localization with the largest count
#     clean_fiducials = [sub_df.sort_values('amp', ascending=False).groupby('frame').first()
#                        for sub_df in fiducials_dfs if len(sub_df) > min_num_frames]
    return fiducials_dfs


def prune_peaks(peaks, radius):
        """
        Pruner method takes blobs list with the third column replaced by
        intensity instead of sigma and then removes the less intense blob
        if its within diameter of a more intense blob.

        Parameters
        ----------
        peaks : pandas DataFrame
        diameter : float
            Allowed spacing between blobs

        Returns
        -------
        A : ndarray
            `array` with overlapping blobs removed.
        """

        # make a copy of blobs otherwise it will be changed
        # create the tree
        kdtree = cKDTree(peaks[["y0", "x0"]].values)
        # query all pairs of points within diameter of each other
        list_of_conflicts = list(kdtree.query_pairs(radius))
        # sort the collisions by max amplitude of the pair
        # we want to deal with collisions between the largest
        # blobs and nearest neighbors first:
        # Consider the following sceneario in 1D
        # A-B-C
        # are all the same distance and colliding with amplitudes
        # A > B > C
        # if we start with the smallest, both B and C will be discarded
        # If we start with the largest, only B will be
        # Sort in descending order
        list_of_conflicts.sort(
            key=lambda x: max(peaks.amp.iloc[x[0]], peaks.amp.iloc[x[1]]),
            reverse=True
        )
        # indices of pruned blobs
        pruned_peaks = set()
        # loop through conflicts
        for idx_a, idx_b in list_of_conflicts:
            # see if we've already pruned one of the pair
            if (idx_a not in pruned_peaks) and (idx_b not in pruned_peaks):
                # compare based on amplitude
                if peaks.amp.iloc[idx_a] > peaks.amp.iloc[idx_b]:
                    pruned_peaks.add(idx_b)
                else:
                    pruned_peaks.add(idx_a)
        # return pruned dataframe
        return peaks.iloc[[
            i for i in range(len(peaks)) if i not in pruned_peaks]
        ]

def palm_hist(df, yx_shape, subsample=1, zrange=None, zsubsample=None):
    bins = [np.arange(s + subsample, step=subsample) - subsample / 2 for s in yx_shape]
    # ungrouped 2d histogram to find peaks, ungrouped to make beads really stand out
    keys = ["y0", "x0"]
    histtype = lambda *args, **kwargs: [np.histogramdd(*args, **kwargs)[0].astype(int)]
    if zsubsample is not None:
        if zrange is None:
            zrange = df.z0.min(), df.z0.max()
        keys = ["z0"] + keys
        bins = [np.arange(zrange[0], zrange[1] + zsubsample, step=zsubsample)] + bins
        histtype = fast_hist3d
    return histtype(df[keys].values, bins=bins)[0]


def find_fiducials(df, yx_shape, subsampling=1, diagnostics=False, **kwargs):
    """Find fiducials in pointilist PALM data
    
    The key here is to realize that there should be on fiducial per frame"""
    # incase we subsample the frame number
    num_frames = df.frame.max() - df.frame.min()
    hist_2d = palm_hist(df, yx_shape, subsampling)
    pf = PeakFinder(hist_2d.astype(int), 1)
    pf.blob_sigma = 1/subsampling
    # no blobs found so try again with a lower threshold
    pf.thresh = 0
    pf.find_blobs()
    blob_thresh = max(threshold_otsu(pf.blobs[:, 3]), num_frames / 10)
    if not pf.blobs.size:
        # still no blobs then raise error
        raise RuntimeError("No blobs found!")
    pf.blobs = pf.blobs[pf.blobs[:,3] > blob_thresh]
    if pf.blobs[:, 3].max() < num_frames * subsampling / 2:
        print("Warning, drift maybe too high to find fiducials")
    if diagnostics:
        pf.plot_blobs(**kwargs)
    # correct positions for subsampling
    return pf.blobs[:, :2] * subsampling


def remove_fiducials(df, yx_shape, df2=None, exclusion_radius=1, **kwargs):
    """Remove fiducials by first finding them in a histogram of localizations and then
    removing all localizations with in exclusion radius of the found ones"""
    if df2 is None:
        df2 = df
    blobs = find_fiducials(df, yx_shape, **kwargs)
    df3 = filter_fiducials(df2, blobs, exclusion_radius)
    # plt.matshow(np.histogramdd(df3[["Y Position", "X Position"]].values, bins=(ybins, xbins))[0], vmax=10, cmap="inferno")
    return df3


# def show_frame_and_mode(lazy_data, frame_num=-1):
#     """Show a given frame with a histogram of values and the mode"""
#     frame = v[frame_num].compute()
#     fig, (ax_im, ax_ht) = plt.subplots(1, 2, figsize=(8, 4))
#     ax_im.matshow(frame, vmax=300, cmap="inferno")
#     ax_im.grid("off")
#     ax_im.set_title(k)
#     ax_ht.hist(frame.ravel(), bins=np.logspace(2, 3, 128), log=True)
#     mode = np.bincount(frame[:128, -128:].ravel()).argmax()
#     ax_ht.axvline(mode, c="r")
#     ax_ht.set_title("Mode = {}".format(mode))
#     ax_ht.set_xscale("log")
#     return fig, (ax_im, ax_ht)


class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class memoize(object):
    """cache the return value of a method

    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.

    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method:
    class Obj(object):
        @memoize
        def add_to(self, arg):
            return self + arg
    Obj.add_to(1) # not enough arguments
    Obj.add_to(1, 2) # returns 3, result is not cached
    """

    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        pfunc = partial(self, obj)
        pfunc.__doc__ = self.func.__doc__
        return pfunc

    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res


# Lazy functions for raw data handling
lazy_imread = dask.delayed(tif.imread, pure=True)

def make_lazy_data(paths):
    """Make a lazy data array from a set of paths to data

    Assumes all data is of same shape and type."""
    lazy_data = [lazy_imread(path) for path in paths]
    # read first image for shape
    sample = tif.imread(paths[0])
    data = [dask.array.from_delayed(ld, shape=sample.shape, dtype=sample.dtype) for ld in lazy_data]
    data_array = dask.array.concatenate(data)
    return data_array

class RawImages(object):
    """A container for lazy raw images"""
    
    def __init__(self, paths_to_raw):
        self.raw = make_lazy_data(paths_to_raw)

    def __len__(self):
        return len(self.raw)

    def make_fiducial_mask(self, frames=slice(-100, None), iters=2,
                           diagnostics=False, dilation_kwargs=None, **kwargs):
        """Make a mask for the fiducials"""
        if dilation_kwargs is None:
            dilation_kwargs = dict(iterations=5)
        # get the median last frames
        # last_frames = last_frames_temp = np.median(self.raw[frames], 0)
        # last_frames = last_frames_temp = self.mean_img
        init_mask = self.mean_img > threshold_triangle(self.mean_img)
        # for i in range(iters):
        #     init_mask = last_frames_temp > threshold_triangle(last_frames_temp)
        #     last_frames_temp[~init_mask] = mode(last_frames_temp.astype(int))
        
        # the beads/fiducials are high, so we want to negate here
        mask = ~ndi.binary_dilation(init_mask, **dilation_kwargs)
        
        if diagnostics:
            plot_kwargs = dict(norm=PowerNorm(0.25), vmax=1000, vmin=100, cmap="inferno")
            if isinstance(diagnostics, dict):
                plot_kwargs.update(diagnostics)
            fig, (ax0, ax1) = plt.subplots(2, figsize=(4, 8))
            ax0.matshow(self.mean_img, **plot_kwargs)
            ax1.matshow(mask * self.mean_img, **plot_kwargs)
            ax0.set_title("Unmasked")
            ax1.set_title("Masked")
            for ax in (ax0, ax1):
                ax.grid("off")
                ax.axis("off")
        
        # we just made the mask so we should recompute the masked_mean if requested again
        try:
            del self.masked_mean
        except AttributeError:
            # masked_mean doesn't exist yet so don't worry
            pass
        
        self.mask = mask


    @cached_property
    def shape(self):
        return self.raw.shape

    @cached_property
    def mean(self):
        """return the mean of lazy_data"""
        return self.raw.astype(float).mean((1, 2)).compute(get=dask.multiprocessing.get)

    @cached_property
    def mean_img(self):
        """return the mean of lazy_data"""
        return self.raw.astype(float).mean(0).compute(get=dask.multiprocessing.get)

    @cached_property
    def masked_mean(self):
        """return the masked mean"""
        raw_reshape = self.raw.reshape(self.raw.shape[0], -1)
        raw_masked = raw_reshape[:, self.mask.ravel()]
        return raw_masked.mean(1).compute(get=dask.multiprocessing.get)

    @property
    def raw_sum(self):
        return self.mean * np.prod(self.raw.shape[1:])

    @memoize
    def raw_bg(self, num_frames=100, masked=False):
        """Take the median of the last num_frames and compute the mean of the
        median frame to estimate the background and bead contributions"""
        if masked:
            try:
                s = self.mask
            except AttributeError:
                self.make_fiducial_mask()
                s = self.mask
        else:
            s = slice(None)
        self.median_frames = np.median(self.raw[-num_frames:], 0)
        return self.median_frames[s].mean()


def auto_z(palm_df, min_dist=0, max_dist=np.inf, nbins=256, diagnostics=False):
    """Automatically find good limits for z"""
    # make histogram
    hist, bins = np.histogram(palm_df.z0, bins=nbins)
    # calculate center of bins
    z = np.diff(bins)/2 +bins[:-1]
    # find the max z
    max_z = z[hist.argmax()]
    # calculate where the first derivative changes sign
    grad = np.gradient(np.sign(np.gradient(hist)))
    if diagnostics:
        fig, (ax0, ax1, ax2) = plt.subplots(3, figsize=(4, 8), sharex=True)
        ax0.fill_between(z, hist, interpolate=False)
        ax1.plot(z, np.gradient(hist))
        ax2.plot(z, grad)
    # where the sign goes from negative to positive
    # is where minima are located
    concave = np.where(grad ==1)[0]
    # order minima according to distance from max
    i = np.diff(np.sign((z[concave] - max_z))).argmax()
    # the two closest are our best bets
    z_mins = z[concave[i:i+2]]
    # enforce limits
    z_mins_min = np.array((-max_dist, min_dist)) + max_z
    z_mins_max = np.array((-min_dist, max_dist)) + max_z
    z_mins = np.clip(z_mins, z_mins_min, z_mins_max)
    # a diagnostic graph for testing
    if diagnostics:
        for zz in z_mins:
            ax0.axvline(zz, color="r")
            
    return z_mins


class PALMData(object):
    """A simple class to manipulate peakselector data"""
    # columns we want to keep


    peak_col = {
        'X Position': "x0",
        'Y Position': "y0",
        '6 N Photons': "nphotons",
        'Frame Number': "frame",
        'Sigma X Pos Full': "sigma_x",
        'Sigma Y Pos Full': "sigma_y",
        'Sigma Z': "sigma_z",
        'Z Position': 'z0',
        'Offset': 'offset',
        'Amplitude': 'amp'
    }

    group_col = {
        'Frame Number': 'frame',
        'Group X Position': 'x0',
        'Group Y Position': 'y0',
        'Group Sigma X Pos': 'sigma_x',
        'Group Sigma Y Pos': 'sigma_y',
        'Sigma Z': "sigma_z",
        'Group N Photons': 'nphotons',
        '24 Group Size': 'groupsize',
        'Group Z Position': 'z0',
        'Offset': 'offset',
        'Amplitude': 'amp'
    }

    def __init__(self, path_to_sav, verbose=False):
        """To initialize the experiment we need to know where the raw data is
        and where the peakselector processed data is
        
        Assumes paths_to_raw are properly sorted"""
        
        # load peakselector data
        raw_df = peakselector_df(path_to_sav, verbose=verbose).astype(float)
        # convert Frame number to int
        raw_df[['Frame Number', '24 Group Size']] = raw_df[['Frame Number', '24 Group Size']].astype(int)
        self.processed = raw_df[list(self.peak_col.keys())]
        self.grouped = grouped_peaks(raw_df)[list(self.group_col.keys())]
        # normalize column names
        self.processed = self.processed.rename(columns=self.peak_col)
        self.grouped = self.grouped.rename(columns=self.group_col)

    def filter_peaks(self, offset=1000, sigma_max=3, nphotons=0, groupsize=5000):
        """Filter internal dataframes"""
        for df_title in ("processed", "grouped"):
            df = self.__dict__[df_title]
            filter_series = (
                (df.offset > 0) & # we know that offset should be around this value.
                (df.offset < offset) &
                (df.sigma_x < sigma_max) &
                (df.sigma_y < sigma_max) &
                (df.nphotons > nphotons)
            )
            if "groupsize" in df.keys():
                filter_series &= df.groupsize < groupsize
            self.__dict__[df_title + "_filtered"] = df[filter_series]

    def hist(self, data_type="grouped", filtered=False):
        if data_type == "grouped":
            if filtered:
                df = self.grouped_filtered
            else:
                df = self.grouped
        elif data_type == "processed":
            if filtered:
                df = self.processed_filtered
            else:
                df = self.processed
        else:
            raise TypeError("Data type {} is of unknown type".format(data_type))
        return df[['offset', 'amp', 'xpos', 'ypos', 'nphotons',
            'sigmax', 'sigmay', 'zpos']].hist(bins=128, figsize=(12, 12), log=True)

    def filter_z(self, min_z=None, max_z=None, **kwargs):
        """Crop the z range of data"""
        if len(self.grouped):
            df = self.grouped
        else:
            df = self.processed
        auto_min_z, auto_max_z = auto_z(df, **kwargs)
        if min_z is None:
            min_z = auto_min_z
        if max_z is None:
            max_z = auto_max_z
        for df_title in ("processed", "grouped"):
            sub_title = "_filtered"
            df = self.__dict__[df_title + sub_title]
            filt = (df.z0 < max_z) & (df.z0 > min_z)
            self.__dict__[df_title + sub_title] = df[filt]

    def filter_peaks_and_beads(self, peak_kwargs, fiducial_kwargs, filter_z_kwargs):
        """Filter individual localizations and remove fiducials"""
        self.filter_peaks(**peak_kwargs)
        self.remove_fiducials(**fiducial_kwargs)
        self.filter_z(**filter_z_kwargs)

    def remove_fiducials(self, yx_shape, subsampling=1, exclusion_radius=1, **kwargs):
        # use processed to find fiducials.
        for df_title in ("processed_filtered", "grouped_filtered"):
            self.__dict__[df_title] = remove_fiducials(self.processed, yx_shape, self.__dict__[df_title],
                                  exclusion_radius=exclusion_radius, **kwargs)

    @cached_property
    def raw_frame(self):
        """Make a groupby object that is by frame"""
        return self.processed.groupby("frame")

    @cached_property
    def raw_counts(self):
        """Number of localizations per frame, not filtering"""
        return self.raw_frame.x0.count()


    def sigmas(self, filt="_filtered", frame=0):
        """Plot sigmas"""
        fig, axs = plt.subplots(2, 3, figsize=(3*4, 2*4), sharex="col")
        
        for sub_axs, dtype in zip(axs, ("processed", "grouped")):
            df = self.__dict__[dtype + filt]
            df = df[df.frame > frame]
            for ax, attr, mult in zip(sub_axs, ("sigma_x", "sigma_y", "sigma_z"), (130, 130, 1)):
                dat = (df[attr] * mult)
                dat.hist(ax=ax, bins="auto", normed=True)
                ax.set_title("{} $\{}$".format(dtype.capitalize(), attr))
                add_line(ax, dat)
                add_line(ax, dat, np.median, color=ax.lines[-1].get_color(), linestyle="--")
                ax.legend(loc="best", frameon=True)
                ax.set_yticks([])
        
        for ax in sub_axs:
            ax.set_xlabel("Precision (nm)")
        
        fig.tight_layout()
        
        return fig, axs

    def photons(self, filt="_filtered", frame=0):
        fig, axs = plt.subplots(2, sharex=True, figsize=(4, 8))
        for ax, dtype in zip(axs, ("processed", "grouped")):
            df = self.__dict__[dtype + filt]
            series = df[df.frame > frame]["nphotons"]
            bins = np.logspace(np.log10(series.min()), np.log10(series.max()), 128)
            series.hist(ax=ax, log=True, bins=bins)
            ax.set_xscale("log")
            # mean line
            add_line(ax, series)
            # median line
            add_line(ax, series, np.median, color=ax.lines[-1].get_color(), linestyle="--")
            ax.legend(frameon=True, loc="best")
            ax.set_title(dtype.capitalize())
        ax.set_xlabel("# of Photons")
        fig.tight_layout
        return fig, axs


def exponent(xdata, amp, rate, offset):
    """Utility function to fit nonlinearly"""
    return amp * np.exp(rate * xdata) + offset


class Data405(object):
    
    def __init__(self, path):
        self.data = pd.read_csv(path, index_col=0, parse_dates=True)
        self.data = self.data.rename(columns={k:k.split(" ")[0].lower() for k in self.data.keys()})
        self.data['date_delta'] = (self.data.index - self.data.index.min())  / np.timedelta64(1,'D')

    def fit(self, lower_limit):
        # we know that the laser is subthreshold below 0.45 V and the max is 5 V, so we want to limit the data between these two
        data_df = self.data
        data_df_crop = data_df[(data_df.reactivation > lower_limit) & (data_df.reactivation < 5)].dropna()
        self.popt, self.pcov = curve_fit(exponent, *data_df_crop[["date_delta", "reactivation"]].values.T)
        data_df["fit"] = exponent(data_df["date_delta"], *self.popt)
        self.fit_win = data_df_crop.index.min(), data_df_crop.index.max()
        
    def plot(self, ax=None, limits=True, lower_limit=0.45):
        # this is fast so no cost
        self.fit(lower_limit)
        if ax is None:
            fig, ax = plt.subplots()
        self.data[["reactivation", "fit"]].plot(ax=ax)
        ax.text(0.1, 0.5, "$y(t) = {:.3f} e^{{{:.3f}t}} + {:.3f}$".format(*self.popt), transform=ax.transAxes)
        if limits:
            for edge in self.fit_win:
                ax.axvline(edge, color="r")


### Fit Functions
def weird(xdata, *args):
    """Honestly this looks like saturation behaviour"""
    res = np.zeros_like(xdata)
    for a, b, c in zip(*(iter(args),) * 3):
        res += a*(1 + b * xdata) ** c
    return res


def stretched_exp(xdata, a, b):
    return a * np.exp(-xdata ** b)


def multi_exp(xdata, *args):
    """Power and exponent"""
    odd = len(args) % 2
    if odd:
        offset = args[-1]
    else:
        offset = 0
    res = np.ones_like(xdata) * offset
    for i in range(0, len(args) - odd, 2):
        a, k = args[i:i + 2]
        res += a * np.exp(-k * xdata)
    return res


class PALMExperiment(object):
    """A simple class to organize our experimental data"""

    def __init__(self, raw_or_paths_to_raw, path_to_sav, path_to_405, verbose=False, init=False,
                 timestep=0.0525, nofeedbackframes=250*1000,**kwargs):
        """To initialize the experiment we need to know where the raw data is
        and where the peakselector processed data is
        
        Assumes paths_to_raw are properly sorted"""
        
        # deal with raw data
        try:
            self.raw = RawImages(raw_or_paths_to_raw)
        except TypeError:
            self.raw = raw_or_paths_to_raw
            
        # load peakselector data
        try:
            self.palm = PALMData(path_to_sav, verbose=verbose)
        except TypeError:
            # assume user has passed PALMData object
            self.palm = path_to_sav

        self.timestep = timestep
        self.nofeedbackframes = nofeedbackframes

        try:
            self.activation = Data405(path_to_405)
        except ValueError:
            self.activation = path_to_405

        if init:
            self.palm.filter_peaks_and_beads(yx_shape=self.raw.shape[-2:])
            self.masked_mean

    def filter_peaks_and_beads(self, peak_kwargs=dict(), fiducial_kwargs=dict(), filter_z_kwargs=dict()):
        """Filter individual localizations and remove fiducials"""
        fiducial_kwargs.update(dict(
            yx_shape=self.raw.shape[-2:],
            cmap="inferno",
            vmax=1000,
            norm=PowerNorm(0.25)
        ))
        self.palm.filter_peaks_and_beads(peak_kwargs, fiducial_kwargs, filter_z_kwargs)


    @cached_property
    def masked_mean(self):
        """The masked mean minus the masked background"""
        return pd.Series(self.raw.masked_mean - self.raw.raw_bg(masked=True),
                         np.arange(len(self.raw.masked_mean))*self.timestep, name="Raw Intensity")

    def plot_drift(self, max_s=0.15, max_fid=10):
        fid_dfs = extract_fiducials(self.palm.processed, find_fiducials(self.palm.processed, self.raw.shape[1:]), 1)
        fid_stats, all_drift = calc_fiducial_stats(fid_dfs)
        fid_stats = fid_stats.sort_values("sigma")
        good_fid = fid_stats[fid_stats.sigma < max_s].iloc[:max_fid]
        if not len(good_fid):
            # if no fiducials survive, just use the best one.
            good_fid = fid_stats.iloc[:1]

        return plot_stats([fid_dfs[g] for g in good_fid.index], z_pix_size=1)


    @property
    def nofeedback(self):
        """The portion of the intensity decay that doesn't have feedback"""
        return self.masked_mean.iloc[:self.nofeedbackframes]

    @property
    def feedback(self):
        """the portion of the intensity decay that does have feedback"""
        return self.masked_mean.iloc[self.nofeedbackframes:]

    def plot_w_fit(self, title, ax=None, func=weird, p0=None):
        if ax is None:
            fig, ax = plt.subplots()
        self.nofeedback.plot(loglog=True, ylim=[1, None], ax=ax)
        # do fit
        if p0 is None:
            if func is weird:
                m = self.nofeedback.max()
                p0 = (m, 1, -1, m / 10, 0.5, -0.5)
                func_label = ("$y(t) = " + "+".join(["{:.3f} (1 + {:.3f} t)^{{{:.3f}}}"] * 2) + "$")

        # do the fit
        popt, pcov = curve_fit(func, self.nofeedback.index, self.nofeedback, p0=p0)
        fit = func(self.nofeedback.index, *popt)
        
        # plot the fit
        ax.plot(self.nofeedback.index, fit,
                label=func_label.format(*popt))

        # save the residuals in case we want them later
        self.resid = self.nofeedback - fit
        
        ax.legend(frameon=True, loc="lower center")
        ax.set_ylabel("Mean Intensity")
        ax.set_xlabel("Time (s)")
        ax.set_title(title)
        return fig, ax

    def plot_all(self):
        fig, axs = plt.subplots(3, figsize=(6, 10))
        (ax0, ax1, ax2) = axs
        ax0.get_shared_x_axes().join(ax0, ax1)
        self.feedback.plot(ax=ax0)
        self.feedback.rolling(1000, 0, center=True).mean().plot(ax=ax0)
        # normalize index
        raw_counts = self.palm.raw_counts.loc[self.nofeedbackframes:]
        raw_counts.index = raw_counts.index * self.timestep
        # plot data and rolling average
        raw_counts.plot(ax=ax1)
        raw_counts.rolling(1000, 0, center=True).mean().plot(ax=ax1)
        self.activation.plot(ax=ax2, limits=False)

        for ax in (ax0, ax1):
            ax.set_xticklabels([])
            ax.set_xlabel("")

        ax0.set_ylabel("Average Frame Intensity\n(Background Subtracted)")
        ax1.set_ylabel("Raw Localizations\nPer Frame (with Fiducials)")
        ax2.set_ylabel("405 Voltage (V)")

        ax0.set_title("Feedback Only")

        fig.tight_layout()
        return fig, (ax0, ax1, ax2)


def add_line(ax, data, func=np.mean, fmt_str=":.0f", **kwargs):
    m = func(data)
    func_name = func.__name__.capitalize()
    # use style colors if available
    if "color" not in kwargs.keys():
        if ax.lines or ax.patches:
            c = next(ax._get_lines.prop_cycler)
        else:
            c = dict(color='r')
        kwargs.update(c)
    ax.axvline(m, label=("{} = {" + fmt_str + "}").format(func_name, m), **kwargs)


def log_bins(data, nbins=128):
    minmax = np.nanmin(data), np.nanmax(data)
    logminmax = np.log10(minmax)
    return np.logspace(*logminmax, num=nbins)


def choose_dtype(max_value):
    """choose the appropriate dtype for saving images"""
    # if any type of float, use float32
    if np.issubdtype(np.inexact, max_value):
        return np.float32
    # check for integers now
    if max_value < 2**8:
        return np.uint8
    elif max_value < 2**16:
        return np.uint16
    elif max_value < 2**32:
        return np.uint32
    return np.float32


def tif_convert(data):
    """convert data for saving as tiff copying if necessary"""
    data = np.asarray(data)
    return data.astype(choose_dtype(data.max()), copy=False)

### Fast histogram stuff

@njit
def jit_hist3d(zpositions, ypositions, xpositions, shape):
    """Generate a histogram of points in 3D

    Parameters
    ----------

    Returns
    -------
    res : ndarray
    """
    nz, ny, nx = shape
    res = np.zeros(shape, np.uint32)
    # need to add ability for arbitraty accumulation
    for z, y, x in zip(zpositions, ypositions, xpositions):
        # bounds check
        if x < nx and y < ny and z < nz:
            res[z, y, x] += 1
    return res


@njit
def jit_hist3d_with_weights(zpositions, ypositions, xpositions, weights,
                            shape):
    """Generate a histogram of points in 3D

    Parameters
    ----------

    Returns
    -------
    res : ndarray
    """
    nz, ny, nx = shape
    res = np.zeros(shape, weights.dtype)
    # need to add ability for arbitraty accumulation
    for z, y, x, w in zip(zpositions, ypositions, xpositions, weights):
        # bounds check
        if x < nx and y < ny and z < nz:
            res[z, y, x] += w
    return res


def fast_hist3d(sample, bins, myrange=None, weights=None):
    """Modified from numpy histogramdd
    Make a 3d histogram, fast, lower memory footprint"""
    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = atleast_2d(sample).T
        N, D = sample.shape

    nbin = np.empty(D, int)
    edges = D * [None]
    dedges = D * [None]

    try:
        M = len(bins)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the'
                ' sample x.')
    except TypeError:
        # bins is an integer
        bins = D * [bins]

    # Select range for each dimension
    # Used only if number of bins is given.
    if myrange is None:
        # Handle empty input. Range can't be determined in that case, use 0-1.
        if N == 0:
            smin = np.zeros(D)
            smax = np.ones(D)
        else:
            smin = atleast_1d(np.array(sample.min(0), float))
            smax = atleast_1d(np.array(sample.max(0), float))
    else:
        if not np.all(np.isfinite(myrange)):
            raise ValueError(
                'myrange parameter must be finite.')
        smin = np.zeros(D)
        smax = np.zeros(D)
        for i in range(D):
            smin[i], smax[i] = myrange[i]

    # Make sure the bins have a finite width.
    for i in range(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - .5
            smax[i] = smax[i] + .5

    # avoid rounding issues for comparisons when dealing with inexact types
    if np.issubdtype(sample.dtype, np.inexact):
        edge_dt = sample.dtype
    else:
        edge_dt = float
    # Create edge arrays
    for i in range(D):
        if np.isscalar(bins[i]):
            if bins[i] < 1:
                raise ValueError(
                    "Element at index %s in `bins` should be a positive "
                    "integer." % i)
            nbin[i] = bins[i] + 2  # +2 for outlier bins
            edges[i] = np.linspace(smin[i], smax[i], nbin[i] - 1, dtype=edge_dt)
        else:
            edges[i] = np.asarray(bins[i], edge_dt)
            nbin[i] = len(edges[i]) + 1  # +1 for outlier bins
        dedges[i] = np.diff(edges[i]).min()
        if np.any(np.asarray(dedges[i]) <= 0):
            raise ValueError(
                "Found bin edge of size <= 0. Did you specify `bins` with"
                "non-monotonic sequence?")

    nbin = np.asarray(nbin)

    # Handle empty input.
    if N == 0:
        return np.zeros(nbin - 2), edges

    # Compute the bin number each sample falls into.
    # np.digitize returns an int64 array when it only needs to be uint32
    Ncount = [np.digitize(sample[:, i], edges[i]).astype(np.uint32) for i in range(D)]
    shape = tuple(len(edges[i]) - 1 for i in range(D))  # -1 for outliers

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in range(D):
        # Rounding precision
        mindiff = dedges[i]
        if not np.isinf(mindiff):
            decimal = int(-np.log10(mindiff)) + 6
            # Find which points are on the rightmost edge.
            not_smaller_than_edge = (sample[:, i] >= edges[i][-1])
            on_edge = (np.around(sample[:, i], decimal) ==
                       np.around(edges[i][-1], decimal))
            # Shift these points one bin to the left.
            Ncount[i][np.where(on_edge & not_smaller_than_edge)[0]] -= 1

    # Flattened histogram matrix (1D)
    # Reshape is used so that overlarge arrays
    # will raise an error.
    # hist = zeros(nbin, float).reshape(-1)

    # # Compute the sample indices in the flattened histogram matrix.
    # ni = nbin.argsort()
    # xy = zeros(N, int)
    # for i in arange(0, D-1):
    #     xy += Ncount[ni[i]] * nbin[ni[i+1:]].prod()
    # xy += Ncount[ni[-1]]

    # # Compute the number of repetitions in xy and assign it to the
    # # flattened histmat.
    # if len(xy) == 0:
    #     return zeros(nbin-2, int), edges

    # flatcount = bincount(xy, weights)
    # a = arange(len(flatcount))
    # hist[a] = flatcount

    # # Shape into a proper matrix
    # hist = hist.reshape(sort(nbin))
    # for i in arange(nbin.size):
    #     j = ni.argsort()[i]
    #     hist = hist.swapaxes(i, j)
    #     ni[i], ni[j] = ni[j], ni[i]

    # # Remove outliers (indices 0 and -1 for each dimension).
    # core = D*[slice(1, -1)]
    # hist = hist[core]

    # # Normalize if normed is True
    # if normed:
    #     s = hist.sum()
    #     for i in arange(D):
    #         shape = ones(D, int)
    #         shape[i] = nbin[i] - 2
    #         hist = hist / dedges[i].reshape(shape)
    #     hist /= s

    # if (hist.shape != nbin - 2).any():
    #     raise RuntimeError(
    #         "Internal Shape Error")
    # for n in Ncount:
    #     print(n.shape)
    #     print(n.dtype)
    if weights is not None:
        weights = np.asarray(weights)
        hist = jit_hist3d_with_weights(*Ncount, weights=weights, shape=shape)
    else:
        hist = jit_hist3d(*Ncount, shape=shape)
    return hist, edges

### Gaussian Rendering
_jit_calc_pad = njit(_calc_pad, nogil=True)

@njit(nogil=True)
def _jit_slice_maker(xs1, ws1):
    """Modified from the version in dphutils to allow jitting"""
    if np.any(ws1 < 0):
        raise ValueError("width cannot be negative")
    # ensure integers
    xs = np.rint(xs1).astype(np.int32)
    ws = np.rint(ws1).astype(np.int32)
    # use _calc_pad
    toreturn = []
    for x, w in zip(xs, ws):
        half2, half1 = _jit_calc_pad(0, w)
        xstart = x - half1
        xend = x + half2
        assert xstart <= xend, "xstart > xend"
        if xend <= 0:
            xstart, xend = 0, 0
        # the max calls are to make slice_maker play nice with edges.
        toreturn.append((max(0, xstart), xend))
    # return a list of slices
    return toreturn


def _gauss(yw, xw, y0, x0, sy, sx):
    """Simple normalized 2D gaussian function for rendering"""
    # for this model, x and y are seperable, so we can generate
    # two gaussians and take the outer product
    y, x = np.arange(yw), np.arange(xw)
    amp = 1 / (2 * np.pi * sy * sx)
    gy = np.exp(-((y - y0) / sy) ** 2 / 2)
    gx = np.exp(-((x - x0) / sx) ** 2 / 2)
    return amp * np.outer(gy, gx)

_jit_gauss = njit(_gauss, nogil=True)


def _gen_img_sub(yx_shape, params, mag, multipliers, diffraction_limit):
    """A sub function for actually rendering the images
    Some of the structure is not really 'pythonic' but its to allow JIT compilation

    Parameters
    ----------
    yx_shape : tuple
        The shape overwhich to render the scene
    params : ndarray (M x N)
        An array containing M localizations with data ordered as
        y0, x0, sigma_y, sigma_x
    mag : int
        The magnification factor to render the scene
    multipliers : array (M) optional
        an array of multipliers so that you can do weigthed averages
        mainly to be used for depth coded MIPs
    diffraction_limit : bool
        Controls whether or not there is a lower limit for the localization precision
        This can have better smoothing.

    Returns
    -------
    img : ndarray
        The rendered image
    """
    # hard coded radius, this is really how many sigmas you want
    # to use to render each gaussian
    radius = 5
    yw, xw = yx_shape
    # initialize the image
    img = np.zeros((yw * mag, xw * mag))
    # iterate through all localizations
    for i in range(len(params)):
        # pull parameters
        y0, x0, sy, sx = params[i]
        # adjust to new magnification
        y0, x0, sy, sx = np.array((y0, x0, sy, sx)) * mag
        if diffraction_limit:
            sy, sx = max(sy, 0.5), max(sx, 0.5)
        # calculate the render window size
        width = np.array((sy, sx)) * radius * 2
        # calculate the area in the image
        (ystart, yend), (xstart, xend) = _jit_slice_maker(np.array((y0, x0)), width)
        # adjust coordinates to window coordinates
        y0 -= ystart
        x0 -= xstart
        # generate gaussian
        g = _jit_gauss((yend - ystart), (xend - xstart), y0, x0, sy, sx)
        # weight if requested
        if len(multipliers):
            g *= multipliers[i]
        # update image
        img[ystart:yend, xstart:xend] += g
        
    return img

_jit_gen_img_sub = njit(_gen_img_sub, nogil=True)


def _gen_img_sub_threaded(yx_shape, df, mag, multipliers, diffraction_limit, numthreads=1):
    keys_for_render = ["y0", "x0", "sigma_y", "sigma_x"]
    df = df[keys_for_render]
    length = len(df)
    chunklen = (length + numthreads - 1) // numthreads
    # Create argument tuples for each input chunk
    df_chunks = [df.iloc[i * chunklen:(i + 1) * chunklen] for i in range(numthreads)]
    mult_chunks = [multipliers[i * chunklen:(i + 1) * chunklen] for i in range(numthreads)]
    lazy_result = [dask.delayed(_jit_gen_img_sub)(yx_shape, df_chunk.values, mag, mult, diffraction_limit)
                               for df_chunk, mult in zip(df_chunks, mult_chunks)]
    lazy_result = dask.array.stack([dask.array.from_delayed(l, np.array(yx_shape) * mag, np.float)
        for l in lazy_result])
    return lazy_result.sum(0)
    

def gen_img(yx_shape, df, mag=10, cmap="gist_rainbow", weight="amp", diffraction_limit=False, numthreads=1):
    """Generate a 2D image, optionally with z color coding

    Parameters
    ----------
    yx_shape : tuple
        The shape overwhich to render the scene
    df : DataFrame
        A DataFrame object containing localization data
    mag : int
        The magnification factor to render the scene
    cmap : "hsv"
        The color coding for the z image, if set to None, only 2D
        will be rendered
    weight : str
        The key with which to weight the z image, valid options are
        "amp" and "nphotons"
    numthreads : int
        The number of threads to use during rendering. (Experimental)

    Returns
    -------
    img : ndarray
        If no cmap is specified then the result is a 2D array
        If a cmap is specified then the result is a 3D array where
        the last axis is RGBA. the A channel is just the intensity
        It will not have gamma or clipping applied.
    """
    # Generate the intensity image
    w = 1 / (np.sqrt(2 * np.pi)) / df["sigma_z"]
    img = _gen_img_sub_threaded(yx_shape, df, mag, w.values, numthreads)
    if cmap is not None:
        # calculate the weighting for each localization
        if weight is not None:
            w *= df[weight]
        # normalize z into the range of 0 to 1
        norm_z = scale(df["z0"].values)
        # Calculate weighted colors for each z position
        wz = (w.values[:, None] * matplotlib.cm.get_cmap(cmap)(norm_z))
        # generate the weighted r, g, anb b images
        args = yx_shape, df, mag
        args2 = diffraction_limit, numthreads
        img_wz_r = _gen_img_sub_threaded(*args, wz[:, 0], *args2)
        img_wz_g = _gen_img_sub_threaded(*args, wz[:, 1], *args2)
        img_wz_b = _gen_img_sub_threaded(*args, wz[:, 2], *args2)
        img_w = _gen_img_sub_threaded(*args, w.values, *args2)
        # combine the images and divide by weights to get a depth-coded RGB image
        rgb = dask.array.dstack((img_wz_r, img_wz_g, img_wz_b)) / img_w[..., None]
        # add on the alpha img
        rgba = dask.array.dstack((rgb, img))
        return DepthCodedImage(rgba.compute(), cmap, mag, (df["z0"].min(), df["z0"].max()))
    else:
        # just return the alpha.
        return img.compute()


class DepthCodedImage(np.ndarray):
    """A specialty class to handle depth coded images, especially saving and displaying"""

    def __new__(cls, data, cmap, mag, zrange):
        # from https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(data).view(cls)
        # add the new attribute to the created instance
        obj.cmap = cmap
        obj.mag = mag
        obj.zrange = zrange
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.cmap = getattr(obj, 'cmap', None)
        self.mag = getattr(obj, 'mag', None)
        self.zrange = getattr(obj, 'zrange', None)

    def save(self, savepath):
        """Save data and metadata to a tif file"""
        info_dict = dict(
                cmap=self.cmap,
                mag=self.mag,
                zrange=self.zrange
            )

        tif_kwargs = dict(compress=6, imagej=True, resolution=(self.mag, self.mag),
            metadata=dict(
                # let's stay agnostic to units for now
                unit="pixel",
                # dump info_dict into string
                info=json.dumps(info_dict),
                axes='YXC'
                )
            )

        tif.imsave(fix_ext(savepath, ".tif"), tif_convert(self), **tif_kwargs)

    @classmethod
    def load(cls, path):
        """Load previously saved data"""
        with tif.TiffFile(path) as file:
            data = file.asarray()
            info_dict = json.loads(file.pages[0].imagej_tags["info"])
        return cls(data, **info_dict)

    @property
    def RGB(self):
        """Return the rgb channels of the image"""
        return np.asarray(self)[..., :3]

    @property
    def alpha(self):
        """Return the alpha channel of the image"""
        return np.asarray(self)[..., 3]

    def _norm_data(self, alpha=False, **kwargs):
        """"""
        # power norm will normalize alpha to 0 to 1 after applying
        # a gamma correction and limiting data to vmin and vmax
        pkwargs = dict(gamma=1, clip=True)
        pkwargs.update(kwargs)
        new_alpha = PowerNorm(**pkwargs)(self.alpha)
        if alpha:
            new_data = np.dstack((self.RGB, new_alpha))
        else:
            new_data = self.RGB * new_alpha[..., None]
        return new_data

    def save_color(self, savepath, alpha=False, **kwargs):
        # normalize path name to make sure that it end's in .tif
        ext = os.path.splitext(savepath)[1]
        if ext.lower() == ".tif":
            alpha = False
        norm_data = self._norm_data(alpha, **kwargs)
        img8bit = (norm_data * 255).astype(np.uint8)
        if ext.lower() == ".tif":
            DepthCodedImage(img8bit, self.cmap, self.mag, self.zrange).save(savepath)
        else:
            imsave(savepath, img8bit)
        
    def plot(self, pixel_size=0.13, unit="μm", scalebar_size=None, auto=False, subplots_kwargs=dict(), norm_kwargs=dict()):
        """Make a nice plot of the data, with a scalebar"""
        # make the figure and axes
        fig, ax = plt.subplots(**subplots_kwargs)
        # make the colorbar plot
        cbar = ax.matshow(np.linspace(self.zrange[0], self.zrange[1], 256).reshape(16, 16), cmap=self.cmap)
        # show the color data
        ax.imshow(self.RGB, interpolation=None)
        # normalize the alpha channel
        nkwargs = dict(gamma=1, clip=True)
        nkwargs.update(norm_kwargs)
        new_alpha = PowerNorm(**nkwargs)(self.alpha)
        # if auto is requested perform it
        if auto:
            vdict = auto_adjust(new_alpha)
        else:
            vdict = dict()
        # overlay the alpha channel over the color image
        ax.matshow(new_alpha, cmap=greys_alpha_cm, **vdict)
        # add the colorbar
        fig.colorbar(cbar, label="z ({})".format(unit))
        # add scalebar if requested
        if scalebar_size:
            # make sure the length makes sense in data units
            scalebar_length = scalebar_size * self.mag / pixel_size
            default_scale_bar_kwargs = dict(
                loc='lower left',
                pad=0.5,
                color='white',
                frameon=False,
                size_vertical=scalebar_length / 10,
                fontproperties=fm.FontProperties(size="large", weight="bold")
            )
            scalebar = AnchoredSizeBar(ax.transData,
                                       scalebar_length,
                                       '{} {}'.format(scalebar_size, unit),
                                       **default_scale_bar_kwargs
                                       )
            # add the scalebar
            ax.add_artist(scalebar)
        # remove ticks and spines
        ax.set_axis_off()
        # return fig and ax for further processing
        return fig, ax


@dask.delayed
def _gen_zplane(yx_shape, df, zplane, mag, diffraction_limit):
    """A subfunction to generate a single z plane"""
    # again a hard coded radius
    radius = 5
    # find the fiducials worth rendering
    df_zplane = df[np.abs(df.z0 - zplane) < df.sigma_z * radius]
    # calculate the amplitude of the z gaussian.
    amps = np.exp(-((df_zplane.z0 - zplane) / df_zplane.sigma_z) ** 2 / 2) / (np.sqrt(2 * np.pi) * df_zplane.sigma_z)
    # generate a 2D image weighted by the z gaussian.
    return _jit_gen_img_sub(yx_shape, df_zplane[["y0", "x0", "sigma_y", "sigma_x"]].values, mag, amps.values, diffraction_limit)


def gen_img_3d(yx_shape, df, zplanes, mag, diffraction_limit):
    """Generate a 3D image with gaussian point clouds

    Parameters
    ----------
    yx_shape : tuple
        The shape overwhich to render the scene
    df : DataFrame
        A DataFrame object containing localization data
    zplanes : array
        The planes at which the user wishes to render
    mag : int
        The magnification factor to render the scene"""
    new_shape = tuple(np.array(yx_shape) * mag)
    # print(dask.array.from_delayed(_gen_zplane(df, yx_shape, zplanes[0], mag), new_shape, np.float))
    rendered_planes = [dask.array.from_delayed(_gen_zplane(yx_shape, df, zplane, mag, diffraction_limit), new_shape, np.float)
                                   for zplane in zplanes]
    to_compute = dask.array.stack(rendered_planes)
    return to_compute.compute()


def save_img_3d(yx_shape, df, savepath, zspacing=None, zplanes=None, mag=10, diffraction_limit=False, **kwargs):
    """Generates and saves a gaussian rendered 3D image along with the relevant metadata in a tif stack

    Parameters
    ----------
    yx_shape : tuple
        The shape overwhich to render the scene
    df : DataFrame
        A DataFrame object containing localization data
    savepath : str
        the path to save the file in.
    mag : int
        The magnification factor to render the scene
    """
    # figure out the zplanes to calculate
    if zplanes is None:
        if zspacing is None:
            raise ValueError("zspacing or zplanes must be specified")
        zplanes = np.arange(df.z0.min(), df.z0.max() + zspacing, zspacing)

    # generate the actual image
    img3d = gen_img_3d(yx_shape, df, zplanes, mag, diffraction_limit)
    # save kwargs
    tif_kwargs = dict(resolution=(mag, mag),
        metadata=dict(
            # spacing is the depth spacing for imagej
            spacing=zspacing,
            # let's stay agnostic to units for now
            unit="pixel",
            # we want imagej to interpret the image as a z-stack
            # so set slices to the length of the image
            slices=len(img3d),
            # This information is mostly redundant with "spacing" but is included
            # incase one wanted to render arbitrarily spaced planes.
            labels=["z = {}".format(zplane) for zplane in zplanes],
            axes="ZYX"
            )
        )

    tif_ready = tif_convert(img3d)
    # check if bigtiff is necessary.
    if tif_ready.nbytes / (4 * 1024**3) < 0.95:
        tif_kwargs.update(dict(imagej=True))
    else:
        tif_kwargs.update(dict(imagej=False, compress=6))

    # incase user wants to change anything
    tif_kwargs.update(kwargs)
    # save the tif
    tif.imsave(fix_ext(savepath, ".tif"), tif_ready, **tif_kwargs)

    # return data to user for further processing.
    return img3d


def fix_ext(path, ext):
    if os.path.splitext(path)[1].lower() != ext.lower():
        path += ext
    return path


def measure_peak_widths(y):
    """Measure peak widths in thresholded data. 

    Parameters
    ----------
    y : iterable (ndarray, 1d)
        binary data

    Returns
    -------
    widths : ndarray, 1d
        Measured widths of the peaks.
    """
    d = np.diff(y)
    i = np.arange(len(d))
    rising_edges = i[d > 0]
    falling_edges = i[d < 0]
    # need to deal with all cases
    # same number of edges
    if len(rising_edges) == len(falling_edges):
        if len(rising_edges) == 0:
            return 0
        # starting and ending with peak
        # if falling edge is first we remove it
        if falling_edges.min() < rising_edges.min():
            widths = np.append(falling_edges, i[-1]) - np.append(0, rising_edges)
        else:
            # only peaks in the middle
            widths = falling_edges - rising_edges
    else:
        # different number of edges
        if len(rising_edges) < len(falling_edges):
            # starting with peak
            widths = falling_edges - np.append(0, rising_edges)
        else:
            # ending with peak
            widths = np.append(falling_edges, i[-1]) - rising_edges
    return widths


def count_blinks(offtimes, gap):
    """Count the number of blinkers based on offtimes and a fixed gap

    ontimes = measure_peak_widths((y > 0) * 1
    offtimes = measure_peak_widths((y == 0) * 1

    """
    breaks = np.nonzero(offtimes > gap)[0]
    if breaks.size:
        blinks = [offtimes[breaks[i] + 1:breaks[i+1]] for i in range(breaks.size - 1)]
    else:
        blinks = [offtimes]
    return ([len(blink) for blink in blinks])
