# # PALM Blinking and Decay Analysis
# The purpose of this notebook is to analyze PALM diagnostic data in a consistent way across data sets. 
# The basic experiment being analyzed here is data that has been reactivated and deactivated multiple times.

import gc
import warnings
import numpy as np
import pandas as pd
# regular plotting
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, ListedColormap

# data loading
from scipy.io import readsav
from skimage.external import tifffile as tif

# get multiprocessing support
import dask
import dask.array
from dask.diagnostics import ProgressBar
import dask.multiprocessing

# need to be able to remove fiducials
import tqdm

# Need partial
from functools import partial

# need otsu
from skimage.filters import threshold_triangle

# need ndimage
import scipy.ndimage as ndi

from scipy.optimize import curve_fit
from pyPALM.drift import (remove_xy_mean, calc_drift, calc_fiducial_stats, find_fiducials,
                          extract_fiducials, remove_drift)
from pyPALM.render import gen_img, save_img_3d

from scipy.spatial import cKDTree


greys_alpha_cm = ListedColormap([(i / 255,) * 3 + ((255 - i) / 255,) for i in range(256)])


def peakselector_df(path, verbose=False):
    """Read a peakselector file into a pandas dataframe"""
    if verbose:
        print("Reading {} into memory ... ".format(path))
    sav = readsav(path, verbose=verbose)
    # pull out cgroupparams, set the byteorder to native and set the rownames
    # sav["totalrawdata"] has the raw data, we can use this to get dimensions.
    df = pd.DataFrame(sav["cgroupparams"].byteswap().newbyteorder(), columns=sav["rownames"].astype(str))
    df.totalrawdata = sav["totalrawdata"]
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

def _get_tif_info(path):
    """Get the tifffile shape with the least amount of work"""
    with tif.TiffFile(path) as mytif:
        s = mytif.series[0]
    return dict(shape=s.shape, dtype=s.dtype)

def make_lazy_data(paths, read_all_shapes=False):
    """Make a lazy data array from a set of paths to data

    Assumes all data is of same shape and type."""
    if read_all_shapes:
        data = [dask.array.from_delayed(lazy_imread(path), **_get_tif_info(path)) for path in tqdm.tqdm_notebook(paths)]
        data_array = dask.array.concatenate(data)
    else:
        # read first image for shape
        sample = _get_tif_info(paths[0])
        data = [dask.array.from_delayed(lazy_imread(path), **sample) for path in paths]
        data_array = dask.array.concatenate(data)
    return data_array

class RawImages(object):
    """A container for lazy raw images"""
    
    def __init__(self, paths_to_raw, read_all_shapes=False):
        self.raw = make_lazy_data(paths_to_raw, read_all_shapes=read_all_shapes)

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
        return raw_masked.astype(float).mean(1).compute(get=dask.multiprocessing.get)

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

def max_z(palm_df, nbins=128):
    """Get the most likely z"""
    hist, bins = np.histogram(palm_df.z0, bins=nbins)
    # calculate center of bins
    z = np.diff(bins)/2 + bins[:-1]
    # find the max z
    max_z = z[hist.argmax()]
    return max_z

def auto_z(palm_df, min_dist=0, max_dist=np.inf, nbins=128, diagnostics=False):
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

    def __init__(self, path_to_sav, verbose=False):
        """To initialize the experiment we need to know where the raw data is
        and where the peakselector processed data is
        
        Assumes paths_to_raw are properly sorted"""
        
        self.peak_col = {
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

        self.group_col = {
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

        # load peakselector data
        raw_df = peakselector_df(path_to_sav, verbose=verbose)
        # the dummy attribute won't stick around after casting, so pull it now.
        self.totalrawdata = raw_df.totalrawdata
        raw_df = raw_df.astype(float)
        # convert Frame number to int
        int_cols = ['Frame Number', '24 Group Size', 'Label Set']
        raw_df[int_cols] = raw_df[int_cols].astype(int)
        # don't discard label column if it's being used
        if raw_df["Label Set"].unique().size > 1:
            d = {"Label Set": "label"}
            self.peak_col.update(d)
            self.group_col.update(d)
        # 
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
        try:
            auto_min_z, auto_max_z = auto_z(df, **kwargs)
        except ValueError:
            warnings.warn("`auto_z` has failed, no clipping of z will be done.")
            auto_min_z, auto_max_z = -np.inf, np.inf
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
    def filtered_frame(self):
        """Make a groupby object that is by frame"""
        return self.processed_filtered.groupby("frame")

    @cached_property
    def raw_counts(self):
        """Number of localizations per frame, not filtering"""
        return self.raw_frame[["x0"]].count()

    @cached_property
    def filtered_counts(self):
        """Number of localizations per frame, not filtering"""
        return self.filtered_frame[["x0"]].count()

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
        if ax is None:
            fig, ax = plt.subplots()
        # check if enough data exists to fit
        if (self.data["reactivation"] > lower_limit).sum() > 100:
            # this is fast so no cost
            self.fit(lower_limit)
            self.data[["reactivation", "fit"]].plot(ax=ax)
            ax.text(0.1, 0.5, "$y(t) = {:.3f} e^{{{:.3f}t}} + {:.3f}$".format(*self.popt), transform=ax.transAxes)
            if limits:
                for edge in self.fit_win:
                    ax.axvline(edge, color="r")
        else:
            warnings.warn("Not enough data to fit")
            self.data["reactivation"].plot(ax=ax)


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
                 timestep=0.0525, chunksize=250, **kwargs):
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

        try:
            self.activation = Data405(path_to_405)
        except ValueError:
            self.activation = path_to_405

        self.feedbackframes = chunksize * len(self.activation.data)
        self.nofeedbackframes = self.palm.processed.frame.max() - self.feedbackframes

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
                         np.arange(len(self.raw.masked_mean)) * self.timestep, name="Raw Intensity")

    @property
    def nphotons(self):
        """number of photons per frame calculated from integrated intensity"""
        return self.masked_mean * (self.raw.shape[-1] * self.raw.shape[-2])

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
        func_label_dict = {weird: ("$y(t) = " + "+".join(["{:.3f} (1 + {:.3f} t)^{{{:.3f}}}"] * 2) + "$")}
        func_label = func_label_dict[weird]

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

    def plot_all(self, **kwargs):
        """Plot many statistics for a PALM photophysics expt"""
        # normalize index
        raw_counts = self.palm.raw_counts.loc[self.nofeedbackframes:]
        # number of photons per frame
        nphotons = self.palm.filtered_frame.nphotons.mean()
        # contrast after feedback is enabeled
        contrast = (nphotons.loc[self.nofeedbackframes:] / self.nphotons.max())
        raw_counts.index = raw_counts.index * self.timestep
        contrast.index = contrast.index * self.timestep
        # make the figure
        fig, axs = plt.subplots(4, figsize=(6, 12))
        (ax0, ax1, ax2, ax3) = axs
        # join the axes together
        ax0.get_shared_x_axes().join(*axs[:3])
        for ax, df in zip(axs[:3], (self.feedback, raw_counts, contrast)):
            df.plot(ax=ax)
            df.rolling(1000, 0, center=True).mean().plot(ax=ax)

        self.activation.plot(ax=ax3, limits=False, **kwargs)

        for ax in axs[:3]:
            ax.set_xticklabels([])
            ax.set_xlabel("")

        ax0.set_ylabel("Average Frame Intensity\n(Background Subtracted)")
        ax1.set_ylabel("Raw Localizations\nPer Frame (with Fiducials)")
        ax2.set_ylabel("Contrast Ratio")
        ax3.set_ylabel("405 Voltage (V)")

        ax0.set_title("Feedback Only")

        fig.tight_layout()
        return fig, axs


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
