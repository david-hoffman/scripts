# # PALM Blinking and Decay Analysis
# The purpose of this notebook is to analyze PALM diagnostic data in a consistent way across data sets. 
# The basic experiment being analyzed here is data that has been reactivated and deactivated multiple times.

import gc
import numpy as np
import pandas as pd
# regular plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm

# data loading
from scipy.io import readsav
from skimage.external import tifffile as tif

# get multiprocessing support
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing

# need to be able to remove fiducials
from peaks.peakfinder import PeakFinder
import tqdm

# Need partial
from functools import partial

# need otsu
from skimage.filters import threshold_otsu, threshold_yen

# need ndimage
import scipy.ndimage as ndi

# for fast hist
from numpy.core import atleast_1d, atleast_2d
from numba import njit
from scipy.spatial import cKDTree


def peakselector_df(path, verbose=False):
    """Read a peakselector file into a pandas dataframe"""
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

def palm_hist(df, yx_shape, subsampling=1):
    bins = [np.arange(s + subsampling, step=subsampling) - subsampling / 2 for s in yx_shape]
    # ungrouped 2d histogram to find peaks, ungrouped to make beads really stand out
    return np.histogramdd(df[["y0", "x0"]].values, bins=bins)[0]


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


def show_frame_and_mode(lazy_data, frame_num=-1):
    """Show a given frame with a histogram of values and the mode"""
    frame = v[frame_num].compute()
    fig, (ax_im, ax_ht) = plt.subplots(1, 2, figsize=(8, 4))
    ax_im.matshow(frame, vmax=300, cmap="inferno")
    ax_im.grid("off")
    ax_im.set_title(k)
    ax_ht.hist(frame.ravel(), bins=np.logspace(2, 3, 128), log=True)
    mode = np.bincount(frame[:128, -128:].ravel()).argmax()
    ax_ht.axvline(mode, c="r")
    ax_ht.set_title("Mode = {}".format(mode))
    ax_ht.set_xscale("log")
    return fig, (ax_im, ax_ht)


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
        last_frames = last_frames_temp = np.median(self.raw[frames], 0)
        for i in range(iters):
            init_mask = last_frames_temp > threshold_yen(last_frames_temp)
            last_frames_temp = last_frames_temp * (~init_mask)
        

        
        # the beads/fiducials are high, so we want to negate here
        mask = ~ndi.binary_dilation(init_mask, **dilation_kwargs)
        
        if diagnostics:
            plot_kwargs = dict(norm=PowerNorm(0.25), vmax=1000, vmin=100, cmap="inferno")
            if isinstance(diagnostics, dict):
                plot_kwargs.update(diagnostics)
            fig, (ax0, ax1) = plt.subplots(2, figsize=(4, 8))
            ax0.matshow(last_frames, **plot_kwargs)
            ax1.matshow(mask * last_frames, **plot_kwargs)
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


    @property
    def shape(self):
        return self.raw.shape

    @cached_property
    def mean(self):
        """return the mean of lazy_data"""
        return self.raw.mean((1, 2)).compute(get=dask.multiprocessing.get)

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
        return np.median(self.raw[-num_frames:], 0)[s].mean()


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
        # initialize filtered ones
        self.processed_filtered = None
        self.grouped_filtered = None

    def filter_peaks(self, offset=1000, sigma_max=3, nphotons=0, groupsize=5000):
        """Filter internal dataframes"""
        for df_title in ("processed", "grouped"):
            df = self.__dict__[df_title]
            filter_series = (
                (df.offset > 0) & # we know that offset should be around this value.
                (df.offset < offset) &
                (df.sigmax < sigma_max) &
                (df.sigmay < sigma_max) &
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

    def filter_peaks_and_beads(self, *args, **kwargs):
        """Filter individual localizations and remove fiducials"""
        self.filter_peaks(*args)
        self.remove_fiducials(**kwargs)

    def remove_fiducials(self, yx_shape, subsampling=1, exclusion_radius=1, **kwargs):
        # use processed to find fiducials.
        for df_title in ("processed_filtered", "grouped_filtered"):
            self.__dict__[df_title] = remove_fiducials(self.processed, yx_shape, self.__dict__[df_title],
                                  exclusion_radius=exclusion_radius, **kwargs)


class PALMExperiment(object):
    """A simple class to organize our experimental data"""

    def __init__(self, raw_or_paths_to_raw, path_to_sav, *args, verbose=True, init=False, **kwargs):
        """To initialize the experiment we need to know where the raw data is
        and where the peakselector processed data is
        
        Assumes paths_to_raw are properly sorted"""
        
        # deal with raw data
        try:
            self.raw = RawImages(raw_or_paths_to_raw)
        except TypeError:
            self.raw = raw_or_paths_to_raw
            
        # load peakselector data
        self.palm = PALMData(path_to_sav, verbose=verbose)

        if init:
            self.palm.filter_peaks_and_beads(yx_shape=self.raw.shape[-2:])
            self.raw.mean

    def make_frame_report(self, **kwargs):
        return make_report(self, "Frame", **kwargs)

    def make_group_report(self, **kwargs):
        return make_report(self, "Grouped", **kwargs)


def add_line(ax, data, func=np.mean, c="k", **kwargs):
    m = func(data)
    func_name = func.__name__.capitalize()
    ax.axvline(m, color=c, label="{} = {:.3f}".format(func_name, m), **kwargs)


def log_bins(data, nbins=128):
    minmax = np.nanmin(data), np.nanmax(data)
    logminmax = np.log10(minmax)
    return np.logspace(*logminmax, num=nbins)





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


def measure_peak_widths(y):
    """Assumes binary data"""
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
    breaks = np.nonzero(offtimes > gap)[0]
    if breaks.size:
        blinks = [offtimes[breaks[i] + 1:breaks[i+1]] for i in range(breaks.size - 1)]
    else:
        blinks = [offtimes]
    return ([len(blink) for blink in blinks])