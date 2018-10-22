# # PALM Blinking and Decay Analysis
# The purpose of this notebook is to analyze PALM diagnostic data in a consistent way across data sets.
# The basic experiment being analyzed here is data that has been reactivated and deactivated multiple times.

import gc
import warnings
import datetime
import re
import numpy as np
import pandas as pd
# regular plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, ListedColormap
import matplotlib.lines as mlines

# data loading
from scipy.io import readsav
from skimage.external import tifffile as tif
from skimage.filters import thresholding
from skimage.draw import circle

# get multiprocessing support
import dask
import dask.array
from dask.diagnostics import ProgressBar
import dask.multiprocessing

# need to be able to remove fiducials
import tqdm

# Need partial
from functools import partial

import itertools as itt

# need ndimage
import scipy.ndimage as ndi

import dphplotting as dplt
from dphutils import *
from pyPALM.drift import *
from pyPALM.utils import *
from pyPALM.registration import *
from pyPALM.grouping import *
from pyPALM.render import gen_img, save_img_3d, tif_convert

from scipy.spatial import cKDTree
from scipy.stats import poisson
from scipy.special import gamma, zeta, gammainc, gammaincc

# override any earlier imports
from dphutils.lm import curve_fit
from scipy.optimize import nnls

from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN

from copy import copy

import logging

logger = logging.getLogger()


greys_alpha_cm = ListedColormap([(i / 255,) * 3 + ((255 - i) / 255,) for i in range(256)])

greys_limit = copy(plt.cm.Greys_r)
greys_limit.set_over('r', 1.0)
greys_limit.set_under('b', 1.0)


def peakselector_df(path, verbose=False):
    """Read a peakselector file into a pandas dataframe"""
    if verbose:
        print("Reading {} into memory ... ".format(path))
    sav = readsav(path, verbose=verbose)
    # pull out cgroupparams, set the byteorder to native and set the rownames
    # sav["totalrawdata"] has the raw data, we can use this to get dimensions.
    df = pd.DataFrame(sav["cgroupparams"].byteswap().newbyteorder(), columns=sav["rownames"].astype(str))
    return df, sav["totalrawdata"].shape


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
    for i, (y, x) in enumerate(tqdm.tqdm(blobs, leave=False, desc="Filtering Fiducials")):
        bead_filter = np.sqrt((df.x0 - x) ** 2 + (df.y0 - y) ** 2) > radius
        blob_filt &= bead_filter
        del bead_filter
        if not i % 10:
            gc.collect()
    gc.collect()
    df = df[blob_filt]
    return df


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
        
        This will be part of the standard library starting in 3.8
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
@dask.delayed(pure=True)
def lazy_imread(path):
    with warnings.catch_warnings():
        # ignore warnings
        warnings.simplefilter("ignore")
        try:
            return tif.imread(path)
        except Exception as e:
            # if I die which path is the isssue?
            e.args = (e.args[0] + " path = " + path,) + e.args[1:]
            raise e


def _get_tif_info(path):
    """Get the tifffile shape with the least amount of work"""
    with warnings.catch_warnings():
        # ignore warnings
        warnings.simplefilter("ignore")
        with tif.TiffFile(path) as mytif:
            s = mytif.series[0]
    return dict(shape=s.shape, dtype=s.dtype.name)


def _get_tif_info_all(paths, read_all=False):
    """Get the tifffile shape with the least amount of work

    Assumes all data is of same shape and type."""
    if read_all:
        # reading all the paths is super slow, speed it up with threaded reads
        # seems fair to assume that bottleneck is IO so threads should be fine.
        tif_info = dask.delayed([dask.delayed(_get_tif_info)(path) for path in paths]).compute()
    else:
        # read first image for shape
        tif_info = _get_tif_info(paths[0])
        # extend to be same shape as data
        tif_info = [tif_info] * len(paths)

    return tif_info


def _get_labview_metadata(txtfile):
    """Get metadata from labview generated text files as dictionary

    Currently only gets date"""

    rx_dict = {
        "date": re.compile(r"(?<=Date :\t).+"),
        "num_imgs": re.compile(r"(?<=# of Imgs :\t)\d+"),
        "time_delta": re.compile(r"(?<=Cycle\(s\) :\t)\d+\.\d+")
    }

    process_dict = {
        "date": lambda x: np.datetime64(datetime.datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p')),
        "num_imgs": int,
        "time_delta": lambda x: np.timedelta64(int(1e6 * float(x)), "us")
    }

    with open(txtfile, "r") as f:
        # load all text
        # not efficient but expedient
        full_text = "\n".join(f.readlines())

    # date = datetime.datetime.strptime(datestr.split("\t")[1][:-1], '%m/%d/%Y %I:%M:%S %p')
    match_dict = {k: v.search(full_text) for k, v in rx_dict.items()}
    return {k: process_dict[k](v.group()) for k, v in match_dict.items()}


def make_lazy_data(paths, tif_info):
    """Make a lazy data array from a set of paths to data

    Assumes all data is of same shape and type."""
    data = [dask.array.from_delayed(lazy_imread(path), **info) for info, path in zip(tif_info, paths)]
    data_array = dask.array.concatenate(data)
    return data_array


class RawImages(object):
    """A container for lazy raw images"""

    def __init__(self, raw, metadata, paths):
        """Initialize a raw images object, most users will do this through the `create` and `load` methods"""
        self.raw = raw
        self.metadata = metadata
        self.paths = paths

        # generate time stamps for each image
        times = [
            np.arange(meta["shape"][0]) * meta["time_delta"] + meta["date"]
            for meta in metadata
        ]

        def one_zeros(n):
            """utility function to make an array of True followed by False"""
            result = np.zeros(n, dtype=bool)
            result[0] = True
            return result

        self.first_frames = np.concatenate([one_zeros(meta["shape"][0]) for meta in metadata])

        # generate an index
        self.date_idx = pd.DatetimeIndex(
            data=np.concatenate(times),
            name="Timestamp"
        )

        if not self.date_idx.is_monotonic:
            logger.warn("Raw data's time index isn't monotonic")

        assert len(self.date_idx) == len(self.raw), "Date index and Raw data don't match, {}, {}".format(self.date_idx, self.raw)

    def __repr__(self):
        """"""
        return (
            "RawImages:\n" +
            "   paths: {} ... {}\n".format(self.paths[0], self.paths[-1]) +
            "   Timestamps: {} ... {}".format(self.date_idx[0], self.date_idx[-1])
        )

    @classmethod
    def create(cls, paths_to_raw, read_all=False):
        """Create a RawImages object from paths to tif files"""
        tif_info = _get_tif_info_all(paths_to_raw, read_all=read_all)
        raw = make_lazy_data(paths_to_raw, tif_info)

        # get date times
        def switch_to_txt(path):
            """switch from tif path to txt path"""
            dname, fname = os.path.split(path)
            split = fname.split("_")
            prefix = "_".join(split[:4])
            return os.path.join(dname, prefix + "_Settings.txt")

        metadata_paths = map(switch_to_txt, paths_to_raw)
        metadata = map(lambda x: _get_labview_metadata(x), metadata_paths)

        # update meta data with tif data
        assert len(tif_info) == len(paths_to_raw), "Paths and meta data don't match"
        metadata = [{**x, **y} for x, y in zip(metadata, tif_info)]

        return cls(raw, metadata, paths_to_raw)

    @classmethod
    def load(cls, fname):
        """Load everything from a pandas managed hdf5 container"""
        metadata = pd.read_hdf(fname, "metadata").to_dict("records")
        for m in metadata:
            m["date"] = np.datetime64(m["date"])
        paths = pd.read_hdf(fname, "paths").tolist()
        # reform tif_info list
        tif_info = [dict(shape=m["shape"], dtype=m["dtype"]) for m in metadata]
        # make dask.Array
        raw = make_lazy_data(paths, tif_info)
        # pass to initilizer
        return cls(raw, metadata, paths)

    def save(self, fname):
        """Dump everything into a pandas managed HDF5 container"""
        save_dict = dict(
            metadata=pd.DataFrame(self.metadata),
            paths=pd.Series(self.paths)
        )
        for k, v in save_dict.items():
            v.to_hdf(fname, k)

    @property
    def shapes(self):
        """shapes of the underlying data files"""
        return np.array([m["shape"] for m in self.metadata])

    @property
    def lengths(self):
        """shapes of the underlying data files"""
        return self.shapes[:, 0]

    def __len__(self):
        return len(self.raw)

    @property
    def shape(self):
        return self.raw.shape

    @cached_property
    def mean(self):
        """return the mean of lazy_data"""
        mymean = self.raw.mean((1, 2)).compute()
        return pd.Series(data=mymean, index=self.date_idx, name="raw_mean")

    @cached_property
    def mean_img(self):
        """return the mean of lazy_data"""
        return self.raw.mean(0).compute()

    @property
    def sum(self):
        return self.mean * np.prod(self.raw.shape[1:])


def max_z(palm_df, nbins=128):
    """Get the most likely z"""
    hist, bins = np.histogram(palm_df.z0, bins=nbins)
    # calculate center of bins
    z = np.diff(bins) / 2 + bins[:-1]
    # find the max z
    max_z = z[hist.argmax()]
    return max_z


def auto_z(palm_df, min_dist=0, max_dist=np.inf, nbins=128, diagnostics=False):
    """Automatically find good limits for z"""
    # make histogram
    hist, bins = np.histogram(palm_df.z0, bins=nbins)
    # calculate center of bins
    z = np.diff(bins) / 2 + bins[:-1]
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
    concave = np.where(grad == 1)[0]
    # order minima according to distance from max
    i = np.diff(np.sign((z[concave] - max_z))).argmax()
    # the two closest are our best bets
    z_mins = z[concave[i:i + 2]]
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

    save_list = "processed", "drift", "drift_corrected", "grouped"

    def __init__(self, shape, *, processed=None, drift=None, drift_corrected=None, grouped=None):
        """"""
        assert len(shape) == 2
        self.shape = shape
        self.processed = processed
        self.drift = drift
        self.drift_corrected = drift_corrected
        self.grouped = grouped

    def __repr__(self):
        """"""
        internal_params = []
        temp = "No data"
        for name in self.save_list:
            attr = getattr(self, name)
            internal_params.append((name, attr is not None))
            if attr is not None and name != "drift":
                temp = attr.columns
        return ("PALMData with shape = {}\n".format(self.shape) +
                "\n".join(["{} ---> {}".format(k, v) for k, v in internal_params]) +
                "\nColumns ---> {}".format(temp)
                )

    @classmethod
    def load_sav(cls, path_to_sav, verbose=False, processed_only=False, include_width=False):
        """To initialize the experiment we need to know where the raw data is
        and where the peakselector processed data is

        Assumes paths_to_raw are properly sorted

        array(['Offset', 'Amplitude', 'X Position', 'Y Position', 'X Peak Width',
               'Y Peak Width', '6 N Photons', 'ChiSquared', 'FitOK',
               'Frame Number', 'Peak Index of Frame', '12 X PkW * Y PkW',
               'Sigma X Pos rtNph', 'Sigma Y Pos rtNph', 'Sigma X Pos Full',
               'Sigma Y Pos Full', '18 Grouped Index', 'Group X Position',
               'Group Y Position', 'Group Sigma X Pos', 'Group Sigma Y Pos',
               'Group N Photons', '24 Group Size', 'Frame Index in Grp',
               'Label Set', 'XY Ellipticity', 'Z Position', 'Sigma Z',
               'XY Group Ellipticity', 'Group Z Position', 'Group Sigma Z'],
              dtype='<U20')
        """

        # add gaussian widths

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
            'Amplitude': 'amp',
            'ChiSquared': "chi2"
        }

        if include_width:
            peak_col.update(
                {
                    'X Peak Width': "width_x",
                    'Y Peak Width': "width_y",
                }
            )

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
            'Amplitude': 'amp',
            'ChiSquared': "chi2"
        }

        # load peakselector data
        raw_df, shape = peakselector_df(path_to_sav, verbose=verbose)
        # don't discard label column if it's being used
        int_cols = ['frame']
        if raw_df["Label Set"].unique().size > 1:
            d = {"Label Set": "label"}
            int_cols += ['label']
            peak_col.update(d)
            group_col.update(d)
        # convert to float
        processed = raw_df[list(peak_col.keys())].astype(float)
        # normalize column names
        processed = processed.rename(columns=peak_col)
        processed[int_cols] = processed[int_cols].astype(int)

        if not processed_only:
            int_cols += ['groupsize']
            grouped = grouped_peaks(raw_df)[list(group_col.keys())].astype(float)
            grouped = grouped.rename(columns=group_col)
            grouped[int_cols] = grouped[int_cols].astype(int)
        else:
            grouped = None

        # collect garbage
        gc.collect()

        return cls(shape, processed=processed, grouped=grouped)

    def drift_correct(self, sz=25, all_frames=True, **kwargs):
        """Run automatic drift correction algorithm, see pyPALM.drift.remove_all_drift for details

        sz is the sigma z cutoff

        all_frames indicates whether you want the frames initialized or not."""
        if all_frames:
            frames_index = pd.RangeIndex(self.processed.frame.min(), self.processed.frame.max() + 1, name="frame")
        else:
            frames_index = None

        _, self.drift, _, self.drift_fiducials = remove_all_drift(
            self.processed[self.processed.sigma_z < sz],
            self.shape, None, frames_index,
            **kwargs
        )
        self.drift_corrected = remove_drift(self.processed, self.drift)
        calc_fiducial_stats(self.drift_fiducials, diagnostics=True)

    def group(self, r, zscaling=10):
        """group the drift corrected data"""
        gap, thresh = check_density(self.shape, self.drift_corrected, 1, dim=3, zscaling=zscaling * 130)
        mygap = int(gap(r))
        self.group_radius = r
        self.group_gap = mygap
        self.group_thresh = thresh
        logger.info("Grouping gap is being set to {}".format(mygap))
        self.grouped = slab_grouper([self.drift_corrected], radius=r, gap=mygap, zscaling=zscaling * 130, numthreads=48)[0]

    def find_fiducials(self, data="drift_corrected", **kwargs):
        """"""
        data = getattr(self, data)
        self.fiducials = find_fiducials(data, self.shape, **kwargs)
        return self.fiducials

    @cached_property
    def fiducials(self):
        return self.find_fiducials("drift_corrected", diagnostics=True)

    def remove_fiducials(self, radius):
        """remove fiducials from data"""
        for df_title in ("drift_corrected", "grouped"):
            try:
                df = getattr(self, df_title)
            except AttributeError:
                continue
            df_nf = filter_fiducials(df, self.fiducials, radius)
            setattr(self, df_title + "_nf", df_nf)
        self.make_fiducial_mask(radius)

    @classmethod
    def load(cls, fname):
        """Load PALMData object from Pandas managed HDF container"""
        shape = tuple(pd.read_hdf(fname, "shape"))
        kwargs = {}
        for obj in cls.save_list:
            try:
                kwargs[obj] = pd.read_hdf(fname, obj)
            except KeyError:
                logger.info("Failed to find {} in {}".format(obj, fname))
                continue
        return cls(shape, **kwargs)

    def save(self, fname):
        """Save object internals into Pandas managed HDF container"""
        pd.Series(self.shape).to_hdf(fname, "shape")
        for obj in self.save_list:
            try:
                getattr(self, obj).to_hdf(fname, obj)
            except AttributeError:
                logger.info("Failed to find {}".format(obj))
                continue

    @cached_property
    def data_mask(self):
        """A mask of the data, i.e. areas that are high density but not fiducials

        Note that this returns True where the data is and False were it isn't, which
        is the opposite of that required for np.ma.masked_array"""
        cimg = gen_img(self.shape, self.grouped_nf, mag=1, hist=True, cmap=None)
        thresh = thresholding.threshold_triangle(cimg)
        return cimg > thresh

    def threshold(self, data="grouped_nf", diagnostics=True):
        """Find a threshold that discriminates between biology and background"""
        # pull internal data structure
        data = getattr(self, data)
        # generate histogram
        cimg = gen_img(self.shape, data, mag=1, hist=True, cmap=None)
        # find threshold
        thresh = thresholding.threshold_triangle(cimg)
        if diagnostics:
            ny, nx = cimg.shape
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8 * nx / ny, 4))

            ax0.matshow(cimg, norm=PowerNorm(0.5), cmap="Greys_r")
            ax0.set_title("Data")
            ax1.matshow(cimg, vmin=thresh, cmap=greys_limit)
            ax1.set_title("Data and Thresholds")
        return thresh

    def make_fiducial_mask(self, radius):
        """A mask at fiducial locations

        Note that this returns True for where the fiducials are and False
        For where they aren't"""
        mask = np.zeros(self.shape)
        for y, x in self.fiducials:
            mask[circle(y, x, radius, shape=self.shape)] = 1
        self.fiducial_mask = mask.astype(bool)

    def hist(self, col="nphotons", **kwargs):
        """Plot # of photons as histograms"""
        def plotter(df, ax):
            """utility function"""
            data = df[col]
            bins = log_bins(data)
            ax = data.hist(bins=bins, log=True, density=True, histtype="stepfilled", ax=ax)
            ax.set_xscale("log")
            # add labels
            mean = data.mean()
            ax.axvline(mean, c="k", label="Mean = {:.2g}".format(mean))
            median = data.median()
            ax.axvline(median, c="k", ls="--", label="Median = {:.2g}".format(median))
            if col == "nphotons":
                mode = data.astype(int).mode().values[0]
                ax.axvline(mode, c="k", ls=":", label="Mode = {:.2g}".format(mode))
            # annotate
            ax.legend()
            return ax

        default_kwargs = dict(figsize=(5,10))
        default_kwargs.update(kwargs)
        fig, all_axs = plt.subplots(3, sharex=True, sharey=False, **default_kwargs)
        (ax_p, ax_s, ax_g) = all_axs

        plotter(self.drift_corrected_nf, ax_p)
        plotter(self.grouped_nf[self.grouped_nf.groupsize == 1], ax_s)
        plotter(self.grouped_nf[self.grouped_nf.groupsize != 1], ax_g)

        ax_p.set_title("Per Frame")
        ax_s.set_title("Grouped Singles")
        ax_g.set_title("Grouped Groups")

        ax_s.set_ylabel("Probability Density")

        ax_g.set_xlabel("# of Photons")

        fig.tight_layout()

        return fig, all_axs

    def sigma_hist(self, xymax=25, zmax=250, pixelsize=130, **kwargs):
        """Plot precisions as histograms"""
        # default kwargs for figsize
        default_kwargs = dict(figsize=(12, 8))
        default_kwargs.update(kwargs)

        # set up figure
        fig, all_axs = plt.subplots(2, 3, sharex="col", **default_kwargs)
        (axs_p, axs_g) = all_axs

        sigmas = ["sigma_x", "sigma_y", "sigma_z"]

        # loop through and start plotting
        for axs, data in zip((axs_p, axs_g), (self.drift_corrected_nf, self.grouped_nf)):
            # convert to nm
            data_nm = data[sigmas] * (pixelsize, pixelsize, 1)
            # make bins
            bins = (np.linspace(0, m, 128) for m in (xymax, xymax, zmax))
            # iterate through data
            for ax, d, b in zip(axs, data_nm, bins):
                dd = data_nm[d]
                filt = dd < b.max()
                dd = dd[filt]
                try:
                    # if grouped plot grouped and not grouped as separate hists
                    groupsize = data.groupsize[filt]
                    g0 = dd[groupsize > 1]
                    g0.hist(bins=b, histtype="stepfilled", ax=ax, zorder=1, alpha=0.7, density=True)
                    median = g0.median()
                    ax.axvline(median, c="k", ls=":", label="{:.1f}".format(median))
                    dd = dd[groupsize == 1]
                except AttributeError:
                    pass
                dd.hist(bins=b, histtype="stepfilled", ax=ax, zorder=0, alpha=0.7, density=True)
                median = dd.median()
                ax.axvline(median, c="k", ls="--", label="{:.1f}".format(median))
                ax.set_yticks([])
                ax.legend()

        # label plots
        for ax, t in zip(axs_p, data_nm):
            ax.set_title("$\\" + t + "$")

        axs_p[0].set_ylabel("Per Frame")
        axs_g[0].set_ylabel("Per Group")

        axs_g[1].set_xlabel("Localization Precision (nm)")

        # update middle legend
        grouped_text, single_text = axs_g[1].legend().get_texts()
        grouped_text.set_text("Grouped Median Precision " + grouped_text.get_text())
        single_text.set_text("Single Median Precision " + single_text.get_text())

        fig.tight_layout()

        return fig, all_axs

    def filter(self, filter_func=None, **kwargs):
        """Filter internal DataFrames, if myfilter is dict, assume that it's
        maximum and minimums for each column, if it's a callable assume it
        takes a DataFrame and returns a DataFrame."""

        # build the filter function
        def filter_func2(df):
            """Wrap both filters into one"""
            if filter_func is not None:
                df = filter_func(df)

            filt_idx = pd.Series(data=np.ones(len(df), dtype=bool), index=df.index)

            for k, v in kwargs.items():
                vmin, vmax = v
                filt_idx &= (vmin < df[k]) & (df[k] < vmax)

            return df[filt_idx]

        to_filt = "processed", "drift_corrected", "grouped"
        new_params = {"shape": self.shape, "drift": self.drift}

        # apply the function to all interal dataframes.
        for param in to_filt:
            df = getattr(self, param, None)
            if df is not None:
                df = filter_func2(df)

            new_params[param] = df

        return self.__class__(**new_params)

    def spatial_filter(self, bin_size=2, low=True):
        """Spatially filter PALM data

        1. Bin grouped data spatially
        2. use spatially binned data to determine a threshold that descriminates between biology and background
        3. Apply that filter to keep biology OR background in ungrouped data
        """

        # step 1 and 2
        thresh = self.threshold("grouped_nf")
        cuts_grouped = [pd.cut(self.grouped_nf[c + "0"], np.arange(0, s + bin_size, bin_size)) for c, s in zip("yx", self.shape)]
        filt = self.grouped_nf.groupby(cuts_grouped).size() < thresh

        if not low:
            filt = ~filt

        def filt_func(df):
            """utility function for filtering dataframe"""
            try:
                return filt[df.name]
            except KeyError as e:
                return False
        # step 3
        cuts = [pd.cut(self.drift_corrected_nf[c + "0"], np.arange(0, s + bin_size, bin_size)) for c, s in zip("yx", self.shape)]
        filtered_palm = self.drift_corrected_nf.groupby(cuts).filter(filt_func)

        return filtered_palm


class Data405(object):
    """An object encapsulating function's related to reactivation data"""
    try:
        calibration = pd.read_excel("//dm11/hesslab/Cryo_data/Data Processing Notebooks/Aggregated 405 Calibration.xlsx")
        from scipy.interpolate import interp1d
        calibrate = interp1d(calibration["voltage"], calibration["mean"])
        calibrated = True
    except FileNotFoundError:
        warnings.warn("Calibration not available ...")

        def calibrate(self, array):
            """Do nothing with input"""
            return array
        calibrated = False

    def __init__(self, path):
        """Read in data, normalize column names and set the time index"""
        self.path = path
        self.data = pd.read_csv(path, index_col=0, parse_dates=True)
        self.data = self.data.rename(columns={k: k.split(" ")[0].lower() for k in self.data.keys()})
        # convert voltage to power
        self.data.reactivation = self.calibrate(self.data.reactivation)
        # calculate date delta in hours
        self.data['date_delta'] = (self.data.index - self.data.index.min()) / np.timedelta64(1, 'h')

    def __repr__(self):
        """A representation of the Data405 Object"""
        return "<Data405({})> Begin activation = {}".format(self.path, self.data.index.min())

    def save(self, fname):
        """"""
        pd.Series(self.path).to_hdf(fname, "path405")

    @classmethod
    def load(cls, fname):
        """"""
        path = pd.read_hdf(fname, "path405").values[0]
        return cls(path)

    def exponent(self, xdata, amp, rate, offset):
        """utility function"""
        return amp * np.exp(rate * xdata) + offset

    def fit(self, lower_limit, upper_limit=np.inf):
        """Fit the internal reactivation data (V or mW) to an exponentially increasing fucntion
        between `lower_limit` and `upper_limit`"""
        # we know that the laser is subthreshold below 0.45 V and the max is 5 V,
        # so we want to limit the data between these two
        data_df = self.data

        # make sure we're in range
        upper_limit = min(upper_limit, data_df.reactivation.max() * 0.999)

        # clip data, dropping previous fit column if it exists
        data_df.drop("fit", axis=1, inplace=True, errors='ignore')
        data_df_crop = data_df[(data_df.reactivation > lower_limit) & (data_df.reactivation < upper_limit)].dropna()

        # fit data
        self.popt, self.pcov = curve_fit(self.exponent, *data_df_crop[["date_delta", "reactivation"]].values.T)
        data_df.loc[data_df_crop.index, "fit"] = self.exponent(data_df_crop["date_delta"], *self.popt)

        self.fit_win = data_df_crop.date_delta.min(), data_df_crop.date_delta.max()

    def plot(self, ax=None, limits=True, lower_limit=None, upper_limit=None):
        """Plot reactivation data on axis between limits"""
        if lower_limit is None:
            lower_limit = self.calibrate(0.45)

        if upper_limit is None:
            upper_limit = np.inf

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        # check if enough data exists to fit
        if (self.data["reactivation"] > lower_limit).sum() > 100:
            # this is fast so no cost
            self.fit(lower_limit, upper_limit)

            equation = "$y(t) = {:.3f} e^{{{:.3f}t}} {:+.3f}$".format(*self.popt)
            tau = r"$\tau = {:.2f}$ hours".format(1 / self.popt[1])
            eqn_txt = "\n".join([equation, tau])

            self.data.plot(x="date_delta", y=["reactivation", "fit"], ax=ax, label=["Data", eqn_txt])

            if limits:
                for i, edge in enumerate(self.fit_win):
                    if i:
                        label = "Fit Limits"
                    else:
                        label = None
                    ax.axvline(edge, color="r", label=label)
        else:
            warnings.warn("Not enough data to fit")
            self.data.plot(x="date_delta", y="reactivation", ax=ax)

        ax.set_xlabel("Time (hours)")
        if self.calibrated:
            ax.set_ylabel("405 nm Power (mW)")
        else:
            ax.set_ylabel("405 nm Voltage")

        ax.legend(loc="best")

        return fig, ax


def weird(xdata, *args):
    """Honestly this looks like saturation behaviour"""
    res = np.zeros_like(xdata)
    for a, b, c in zip(*(iter(args),) * 3):
        res += a * (1 + b * xdata) ** c
    return res


class PALMExperiment(object):
    """A simple class to organize our experimental data"""

    def __init__(self, raw, palm, activation, cached_data=None):
        """To initialize the experiment we need raw data, palm data and activation
        each of these are RawImages, PALMData and Data405 objects respectively"""

        # deal with raw data
        self.raw = raw
        self.palm = palm
        self.activation = activation

        # time at which activation starts
        self.activation_start = self.activation.data.index.min()

        # new time index with 0 at start of activation
        self.time_idx = (self.raw.date_idx - self.activation_start) / np.timedelta64(1, 'h')

        # frame at which activation begins
        self.frame_start = np.abs(self.time_idx).argmin()

        if cached_data is None:
            self.cached_data = pd.DataFrame(index=self.time_idx)
        else:
            self.cached_data = cached_data
            self.cached_data.index = self.time_idx

        self.output = dict()
        # active_pixel_density=palm.data_mask.sum() / palm.data_mask.size

    def __repr__(self):
        """A representation of this class"""
        sub_reps = ("\n" + "+" * 80 + "\n").join([repr(self.raw), repr(self.palm), repr(self.activation), repr(self.cached_data.columns)])
        top_str = "This PALM Experiment consists of:\n" + "=" * 80 + "\n"
        return top_str + sub_reps

    @classmethod
    def load(cls, fname):
        """Load data from a pandas managed HDF5 store"""
        raw = RawImages.load(fname)
        palm = PALMData.load(fname)
        activation = Data405.load(fname)
        try:
            cached_data = pd.read_hdf(fname, "cached_data")
        except KeyError:
            logger.info("No cached data found")
            cached_data = None
        return cls(raw, palm, activation, cached_data)

    def save(self, fname):
        """Save data to a pandas managed HDF store"""
        self.raw.save(fname)
        self.palm.save(fname)
        self.activation.save(fname)
        self.cached_data.to_hdf(fname, "cached_data")

    def masked_agg(self, masktype, agg_func=np.median, agg_args=(), agg_kwargs={}, prefilter=False):
        """Mask and aggregate raw data along frame direction with agg_func

        Save results on RawImages object"""
        # make sure our agg_func works with masked arrays
        try:
            agg_func2 = getattr(np.ma, agg_func.__name__)
        except AttributeError:
            logger.info("{} not found".format(agg_func))

            def agg_func2(masked_array, *args, **kwargs):
                array = np.ma.filled(masked_array, np.nan)
                return agg_func(array, *args, **kwargs)

        # we have to reverse the mask for how np.ma.masked_array works
        if masktype.lower() == "data":
            mask = ~self.palm.data_mask
        elif masktype.lower() == "fiducials":
            mask = ~self.palm.fiducial_mask
        elif masktype.lower() == "background":
            mask = (self.palm.fiducial_mask | self.palm.data_mask)
        else:
            raise ValueError("masktype {} not recognized, needs to be one of 'data', 'fiducials' or 'background'".format(masktype))

        # figure out how to split the drift to align with raw data
        cut_points = np.append(0, self.raw.lengths).cumsum()

        @dask.delayed
        def shift_and_mask(chunk, chunked_shifts):
            """Drift correct raw data (based on PALM drift correction), mask and aggregate"""
            # calculate chunked mask
            assert len(chunk) == len(chunked_shifts), "Lengths don't match, {} != {}".format(chunk, chunked_shifts)

            # convert image data to float
            if prefilter:
                # median filter spatially, not temporally
                chunk = ndi.median_filter(chunk, (1, 3, 3))
            chunk = chunk.astype(float)

            # drift correct the chunk
            shifted_chunk = np.array([
                # we're drift correcting the *data* so negative shifts
                # fill with nan's so we can mask those off too.
                ndi.shift(data, (-y, -x), order=1, cval=np.nan)
                for data, (y, x) in zip(chunk, chunked_shifts)
            ])

            # True indicates a masked (i.e. invalid) data.
            # cut out areas that were shifted out of frame
            extended_mask = np.array([mask] * len(shifted_chunk)) | ~np.isfinite(shifted_chunk)

            # mask drift corrected raw data
            shifted_masked_array = np.ma.masked_array(shifted_chunk, extended_mask)

            # return aggregated data along frame axis
            return np.asarray(agg_func2(shifted_masked_array, *agg_args, axis=(1, 2), **agg_kwargs))

        # get x, y shifts from palm drift, interpolating missing values
        shifts = self.palm.drift[["y0", "x0"]].reindex(pd.RangeIndex(len(self.raw))).interpolate("slinear", fill_value="extrapolate", limit_direction="both").values

        # set up computation tree
        to_compute = [
            shift_and_mask(lazy_imread(path), shifts[cut_points[i]:cut_points[i + 1]])
            for i, path in enumerate(self.raw.paths)
        ]

        # get masked results
        masked_result = np.concatenate(dask.delayed(to_compute).compute(scheduler="processes"))

        # save these attributes on the RawImages Object
        attrname = "masked_" + masktype.lower() + "_" + agg_func.__name__
        if prefilter:
            attrname = attrname + "_prefilter"
        logger.info("Setting {}".format(attrname))
        self.cached_data[attrname] = masked_result

        return masked_result

    def calc_all_masked_aggs(self, fname=None):
        """Convenience function to calculate and save the usual masked aggregations."""
        if fname is not None:
            """update store"""
            def save():
                self.cached_data.to_hdf(fname, "cached_data")
        else:
            def save():
                pass

        # calculate things and save if requested.
        self.masked_agg("data")
        save()
        self.masked_agg("data", np.max)
        save()
        self.masked_agg("data", np.max, prefilter=True)
        save()
        self.masked_agg("data", np.nanpercentile, agg_args=(99,))
        save()
        self.masked_agg("data", np.nanpercentile, agg_args=(99,), prefilter=True)
        save()
        self.masked_agg("fiducials")
        save()
        self.masked_agg("background")
        save()

    def agg(self, agg_func=np.median):
        """Aggregate raw data along frame direction with agg_func"""

        @dask.delayed
        def agg_sub(chunk):
            """Calculate agg"""
            return np.asarray(agg_func(chunk, (1, 2)))

        to_compute = dask.delayed([agg_sub(lazy_imread(path)) for path in self.raw.paths])

        result = np.concatenate(to_compute.compute(scheduler="processes"))

        # save these attributes on the RawImages Object
        attrname = masktype.lower() + "_" + agg_func.__name__
        logger.info("Setting {}".format(attrname))
        self.cached_data[attrname] = result

        return result

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

    @property
    def frame(self):
        """Make a groupby object that is by frame"""
        return self.palm.drift_corrected_nf.groupby("frame")

    @cached_property
    def nphotons(self):
        """median number of photons per frame per localization excluding fiducials"""
        nphotons = self.frame.nphotons.median().reindex(pd.RangeIndex(len(self.raw)))
        nphotons.index = self.time_idx
        nphotons = nphotons.sort_index()
        return nphotons

    @cached_property
    def amp(self):
        """median amplitude per frame per localization excluding fiducials"""
        amp = self.frame.amp.median().reindex(pd.RangeIndex(len(self.raw)))
        amp.index = self.time_idx
        amp = amp.sort_index()
        return amp

    @cached_property
    def counts(self):
        """Number of localizations per frame, excluding fiducials"""
        counts = self.frame.size().reindex(pd.RangeIndex(len(self.raw)))
        counts.index = self.time_idx
        counts = counts.sort_index()
        return counts

    @cached_property
    def grouped_counts(self):
        """Number of localizations per frame, excluding fiducials"""
        counts = self.palm.grouped_nf.groupby("frame").size().reindex(pd.RangeIndex(len(self.raw)))
        counts.index = self.time_idx
        counts = counts.sort_index()
        return counts

    @cached_property
    def intensity(self):
        """Median intensity within masked (data) area, in # of photons"""
        # make sure intensity doesn't go to 0, doesn't make physical sense and would cause
        # contrast ratio to explode.
        intensity = np.fmax(self.cached_data.masked_data_median - self.cached_data.masked_background_median, 1)
        return intensity.sort_index()

    @cached_property
    def contrast_ratio(self):
        """Contrast ratio is average number of photons per frame per localization divided
        by average number of photons in data masked area"""
        # contrast after feedback is enabled
        # denominator is average number of photons per pixel
        return self.nphotons / self.intensity

    def plot_feedback(self, **kwargs):
        """Make plots for time when feedback is on"""

        # make the figure
        fig, axs = plt.subplots(4, figsize=(6, 12), sharex=True)

        # end when activation starts
        self._plot_sub(axs[:3], slice(0, None))

        # add activation plot
        self.activation.plot(ax=axs[-1], limits=False, **kwargs)

        # save fits for later
        self.output["popt_react"] = self.activation.popt
        self.output["pcov_react"] = self.activation.pcov
        # set figure title
        axs[0].set_title("Feedback Only")
        axs[0].legend()

        fig.tight_layout()

        return fig, axs

    def plot_all(self, fit_start=None, components=2, **kwargs):
        """Plot entire experiment"""

        # make the figure
        fig, axs = plt.subplots(4, figsize=(6, 12), sharex=True)

        # Plot all data
        self._plot_sub(axs[:3])

        # add activation plot
        self.activation.plot(ax=axs[-1], limits=False, **kwargs)

        # save fits for later
        self.output["popt_react"] = self.activation.popt
        self.output["pcov_react"] = self.activation.pcov

        # add line for t = 0
        for ax in axs:
            ax.axvline(0, color="r")

        if fit_start is not None:
            self._fit_and_add_to_plot(axs[1], fit_start, components)
            axs[1].legend()
        else:
            axs[0].legend()

        fig.tight_layout()

        return fig, axs

    def _fit_and_add_to_plot(self, ax, fit_start, components):
        # pull data to fit
        fit_data = self.counts.loc[fit_start:0].dropna()
        # make index, start from 0
        xdata = (fit_data.index - fit_data.index.min()).values
        # fit the data
        popt, pcov = multi_exp_fit(fit_data.values, xdata, components, offset=True, method="mle")

        # save for later analysis.
        self.output["popt_decay"] = popt
        self.output["pcov_decay"] = pcov

        # make the label
        label_base = "$y(t) = " + "{:+.3g} e^{{-{:.2g}t}}" * (len(popt) // 2) + " {:+.2g}$" * (len(popt) % 2)

        # add the fit line and legend
        ax.plot(fit_data.index, multi_exp(xdata, *popt), label=label_base.format(*popt), ls="--")

    def plot_nofeedback(self, fit_start=None, components=2):
        """Plot many statistics for a PALM photophysics expt"""

        # end at 0 time
        fig, axs = self._plot_sub(None, s=slice(None, 0))

        axs[0].set_title("No Feedback")
        axs[-1].set_xlabel("Time (hours)")

        # add a fit to the counts decay
        # pull axis we want
        if fit_start is not None:
            self._fit_and_add_to_plot(axs[1], fit_start, components)
            axs[1].legend()
        else:
            axs[0].legend()

        fig.tight_layout()

        return fig, axs

    def _plot_sub(self, axs=None, s=slice(None, None, None), no_first=True):
        """Plot many statistics for a PALM photophysics expt"""

        # make the figure
        if axs is None:
            fig, axs = plt.subplots(3, figsize=(6, 9), sharex=True)
        else:
            fig = axs[0].get_figure()

        titles = (
            "Average # of Photons / Pixel",
            "Raw Localizations\nPer Frame (without Fiducials)",
            "Contrast Ratio"
        )

        data = self.intensity, self.counts, self.contrast_ratio

        for ax, df, title in zip(axs, data, titles):
            if no_first:
                df = df[~self.raw.first_frames]
            df = df.loc[s]
            df.plot(ax=ax, label="Raw data")
            df.rolling(1000, 0, center=True).median().plot(ax=ax, label="Rolling mean (1000)")
            ax.set_ylabel(title)

        ax_contrast = axs[2]

        # remove legends
        for l in ax_contrast.lines:
            l.set_label("_")

        # add anotation
        if s.start is None:
            med_pre = self.contrast_ratio.loc[:0].median()
            ax_contrast.hlines(med_pre, self.contrast_ratio.index.min(), 0, "C2", lw=2,
                               label="Pre-activation {:.1f}".format(med_pre), zorder=3)
        if s.stop is None:
            med_post = self.contrast_ratio.loc[0:].median()
            ax_contrast.hlines(med_post, 0, self.contrast_ratio.index.max(), "C3", lw=2,
                               label="Post-activation {:.1f}".format(med_post), zorder=3)

            ax_counts = axs[1]
            # remove legends
            for l in ax_counts.lines:
                l.set_label("_")

            # calculate set point, and store for later retrival.
            set_point = self.set_point = self.counts.loc[0:].median()
            # save set_point for later
            self.output["set_point"] = set_point
            ax_counts.hlines(set_point, 0, self.counts.index.max(), "C2", lw=2,
                             label="Set Point {:}".format(set_point), zorder=3)
            ax_counts.legend()

        ax_contrast.legend()

        return fig, axs

    def plot_contrast(self, s=slice(None), background=True):
        """Plot the various ways of calculating the contrast ratio"""
        # data to use
        titles = {
            "contrast": "Median # Photons / Masked Median",
            "contrast2": "Median Amplitude / Masked Median",
            "masked_data_amax": "Masked Max / Masked Median",
            "masked_data_amax_prefilter": "Filtered Masked Max / Masked Median",
            "masked_data_nanpercentile": "Masked 99% / Masked Median",
            "masked_data_nanpercentile_prefilter": "Filtered Masked 99% / Masked Median"
        }

        if background:
            bg = self.cached_data.masked_background_median
        else:
            bg = 100.0
        # make sure intensity doesn't go below 1, if intensity were zero then
        # contrast ratio would be infinite and it would be very hard to tell
        # the differences between fluorophores.
        intensity = np.fmax(self.cached_data.masked_data_median - bg, 1)

        assert np.array_equal(self.cached_data.index, intensity.index)

        # build data
        data_dict = {}
        for p in titles:
            try:
                # pin numerator to positive
                data_dict[p] = np.fmax((self.cached_data[p] - bg), 0) / intensity
            except KeyError:
                if "contrast" not in p:
                    logger.info("Couldn't find {}".format(p))
                continue

        data_dict["contrast"] = self.nphotons / intensity
        data_dict["contrast2"] = self.amp / intensity

        # setup plot
        num_d = len(data_dict)
        fig, axs = plt.subplots(num_d, 2, sharex="col", sharey="row", gridspec_kw=dict(width_ratios=(3, 1)), figsize=(6, 3 * num_d))

        # make plot
        shared_hist = axs[0, 1].get_shared_x_axes()
        for (ax_plot, ax_hist), (p, contrast) in zip(axs, sorted(data_dict.items())):
            # trim data
            contrast = contrast.loc[s]
            # turn off sharing on the histograms
            shared_hist.remove(ax_hist)
            # turn off x ticking on histograms
            ax_hist.xaxis.set_major_locator(plt.NullLocator())
            ax_hist.xaxis.set_minor_locator(plt.NullLocator())

            # calculate rolling mean
            roll = contrast.rolling(1000, 0, True).median()

            # plot
            for d, label in zip((contrast, roll), ("Data", "Rolling Mean, 1000 Frames")):
                d.plot(ax=ax_plot, label=label, zorder=2)
                d.hist(bins=128, ax=ax_hist, orientation='horizontal', histtype="stepfilled", density=True)

            med_pre = contrast.loc[:0].median()
            med_post = contrast.loc[0:].median()

            # save contrast data for further analysis
            self.output[p] = {"pre": med_pre, "post": med_post}

            # add anotation
            ax_plot.hlines(med_pre, contrast.index.min(), 0, "C2", lw=2,
                           label="Pre-activation {:.1f}".format(med_pre), zorder=3)
            ax_plot.hlines(med_post, 0, contrast.index.max(), "C3", lw=2,
                           label="Post-activation {:.1f}".format(med_post), zorder=3)

            ax_hist.axhline(med_pre, color="C2", linewidth=2)
            ax_hist.axhline(med_post, color="C3", linewidth=2)

            ax_plot.set_title(titles[p])
            ax_plot.legend()

        fig.tight_layout()

        return fig, axs

    def calc_density(self, diagnostics=False):
        """Estimate molecular density from decay of grouped counts"""
        # calculate cumulative counts prior to activation
        pre_counts = self.grouped_counts.loc[:0].cumsum()
        # reset min for fitting.
        pre_counts.index -= pre_counts.index.min()

        try:
            # try fitting to triple exponential (3 was found heuristically)
            popt, pcov = multi_exp_fit(pre_counts.values, pre_counts.index.values, 3)
        except RuntimeError:
            logger.warning("Couldn't estimate density_asymptote")
            popt = [np.nan]

        # make simple plot if requested
        if diagnostics:
            fig, ax = plt.subplots()
            pre_counts.plot(ax=ax, label="Data")
            if len(popt) > 1:
                ax.plot(pre_counts.index, multi_exp(pre_counts.index, *popt), label="Fit")
                ax.axhline(popt[-1], color="C2", label="Density = {:g}".format(popt[-1]))
                ax.legend()

        # save densities as object attributes.
        self.output["density"] = pre_counts.iloc[-1]
        self.output["density_asymptote"] = popt[-1]

        self.output["active_pixel_density"] = self.palm.data_mask.sum() / self.palm.data_mask.size

    def output_to_series(self):
        """Convert the internal output variable to a nicely formated series"""
        output = self.output
        s = pd.Series()
        s["A"], s["ka"], s["B"], s["kb"], s["offset"] = output["popt_decay"]
        s["taua"], s["taub"] = 1 / s.ka, 1 / s.kb

        s["tau_react"] = 1 / output["popt_react"][1]

        s["kDB"] = (s.B * s.ka + s.A * s.kb) / (s.A + s.B)
        s["kBK"] = s.ka * s.kb / s.kDB
        s["kBD"] = (s.A * s.B * (s.ka - s.kb)**2) / ((s.A + s.B) * (s.B * s.ka + s.A * s.kb))

        s["tauDB"], s["tauBK"], s["tauBD"] = s.kDB, s.kBK, s.kBD

        for k in ("set_point", "density", "density_asymptote", "active_pixel_density"):
            s[k] = output[k]

        keys = (
            'contrast',
            'contrast2',
            'masked_data_amax',
            'masked_data_amax_prefilter',
            'masked_data_nanpercentile',
            'masked_data_nanpercentile_prefilter'
        )
        for k in keys:
            for kk in ("pre", "post"):
                s["{}_{}".format(k, kk)] = output[k][kk]
        return s


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
    minmax = np.nanmin(data), np.nanmax(data) + 1
    logminmax = np.log10(minmax)
    return np.logspace(*logminmax, num=nbins)


def measure_peak_widths(trace):
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
    trace = np.asanyarray(trace)
    # deal with all on or off
    if (trace == 1).all():
        logging.debug("All 1s")
        widths = [len(trace)]
    elif (trace == 0).all():
        logging.debug("All 0s")
        widths = [0]
    else:
        d = np.diff(trace)
        i = np.arange(len(d))
        rising_edges = i[d > 0]
        falling_edges = i[d < 0]
        # need to deal with all cases
        # same number of edges
        if len(rising_edges) == len(falling_edges):
            logging.debug("same number of edges")
            if len(rising_edges) == 0:
                return 0
            # starting and ending with peak
            # if falling edge is first we remove it
            if falling_edges.min() < rising_edges.min():
                logging.debug("starting/ending with peak")
                # if trace starts and ends with peak, then we need to add a falling edge
                # at the end of the trace and a rising edge before the beginning
                widths = np.append(falling_edges, i[-1] + 1) - np.append(-1, rising_edges)
            else:
                logging.debug("Peaks in middle")
                # only peaks in the middle
                widths = falling_edges - rising_edges
        else:
            # different number of edges
            logging.debug("different number of edges")
            if len(rising_edges) < len(falling_edges):
                # starting with peak
                logging.debug("starting with peak")
                widths = falling_edges - np.append(0, rising_edges)
                widths[0] += 1
            else:
                # ending with peak
                logging.debug("ending with peak")
                widths = np.append(falling_edges, i[-1] + 1) - rising_edges

    return np.asarray(widths)


def _clear_zeros(a):
    """Remove zeros from an array"""
    a = np.asanyarray(a)
    return a[a > 0]


def make_trace(f, max_frame):
    return f.groupby("frame").size().reindex(np.arange(max_frame)).fillna(0).astype(int).values


def on_off_times(trace, trim=False):
    """Measure on and off times for a trace, triming leading and trailing offtimes if requested"""
    # make sure input is array
    trace = np.asanyarray(trace)

    # calculate on and offtimes
    ontimes = measure_peak_widths((trace > 0) * 1)
    offtimes = measure_peak_widths((trace == 0) * 1)

    # user requested triming
    if trim:
        # if all off then clear arrays
        if offtimes[0] == len(trace):
            ontimes = offtimes = np.array([])
        else:
            # if we begin with 0 then trim first offtime
            if trace[0] == 0:
                start = 1
            else:
                start = 0
            # if we end with zero then trim last offtime
            if trace[-1] == 0:
                end = -1
            else:
                end = None
            offtimes = offtimes[start:end]
    # clear zeros
    return _clear_zeros(ontimes), _clear_zeros(offtimes)


def on_off_times_fast(df):
    """Measure on and off times for a trace, triming leading and trailing offtimes if requested"""
    if not len(df):
        # empty DataFrame
        return array([], dtype="int64"), array([], dtype="int64")
    # find all frames with more than one event
    trace = np.sort(df.frame.unique())
    # calculate the spacing between events
    diff = np.append(1, np.diff(trace))
    # getting off times directly is easy, just look at diff of frames
    # and get rid of differences less than 2, i.e. the molecule
    # has to actually turn off once, then minus 1 to correct for the next
    # time it comes back one
    off = diff[diff > 1] - 1
    # I want to find places where the molecule switches state
    # and sum between these points, so find the break points
    # cumulative sum
    # bincount to get on times
    on = np.bincount((diff > 1).cumsum())
    return on, off


def fast_group(df, gap):
    """Group data assuming that spatial bits have been taken care of"""
    df = df.copy()
    frame_diff = df.frame.diff()
    df["group_id"] = (frame_diff >= gap).cumsum()
    return df


def bhattacharyya(mus, sigmas):
    """Calculate the Bhattacharyya distance between two normal distributions
    with diagonal covariance matrices
    https://en.wikipedia.org/wiki/Bhattacharyya_distance"""
    # convert sigmas in vars
    s1, s2 = sigmas**2
    m1, m2 = mus
    m = m1 - m2
    s = (s1 + s2) / 2
    # we assume covariance is diagnonal
    part1 = 1 / 8 * m ** 2 / s
    part2 = 1 / 2 * np.log(s.prod() / np.sqrt(s1.prod() * s2.prod()))
    return part1.sum() + part2


def make_matrix(df, min_sigma=0, coords="xy"):
    """Calculate distance matrix for df
    
    Each entry is the Bhattacharyya distance between the two
    probability distributions defined by the grouped SMLM events
    """
    
    # make distance matrix
    n = len(df)
    mat = np.zeros((n, n))
    
    # fill it in
    # pull data from DataFrame first, speeds up operations    
    mus = df[[c + "0" for c in coords]].values
    Sigmas = np.fmax(min_sigma, df[["sigma_" + c for c in coords]].values)
    for s in itt.combinations(range(n), 2):
        mat[s] = mat[s[::-1]] = bhattacharyya(mus[s, :], Sigmas[s, :])
    return mat


def cluster_groups(df, *args, affinity="distance", diagnostics=False, **kwargs):
    """Cluster grouped localizations based on distance matrix (Bhattacharyya)"""
    # If there's only one point return
    if len(df) < 2:
        return np.array([len(df)])

    # make the distance matrix
    mat = make_matrix(df, *args, **kwargs)
    
    # choose which metric to use
    if affinity == "distance":
        # amat = np.exp(np.exp(-mat) - 1)
        amat = np.exp(-mat) - 1
    else:
        amat = np.exp(-mat**2)

    amat[~np.isfinite(amat)] = -1e16

    # cluster the data
    aff = AffinityPropagation(affinity="precomputed")
    aff.fit(amat)

    # output diagnostics if wanted
    if diagnostics:
        df_c = df.copy()
        df_c["group_id"] = aff.labels_
        fig, ax = plt.subplots()
        ax.scatter(df_c.x0, df_c.y0, c=df_c.group_id, s=df_c.sigma_z, edgecolor="k", cmap="tab10", vmax=10)

    # return the new number of blinks.
    if not np.isfinite(aff.labels_).all():
        warnings.warn("Clustering failed")
        return np.array([])

    blinks = np.bincount(aff.labels_)
    return blinks


def count_blinks(onofftimes, gap):
    """Count the number of blinkers based on offtimes and a fixed gap"""
    # assume we pass the output of `on_off_times`
    ontimes, offtimes = onofftimes
    # count the number of gaps that are larger than gap - 1
    # this is due to the grouping implementation
    blinks = (offtimes >= gap - 1).sum()
    # check if there are more on times than off times (meaning peaks are at edges)
    diff = len(ontimes) - len(offtimes)
    if diff > 0:
        blinks += diff
    return blinks


def fit_power_law(x, y, maxiters=100, floor=0.1, upper_limit=None, lower_limit=None, include_offset=False):
    """Fit power law to data, iteratively truncating long noisy tail if desired"""
    # initialize iteration variables
    all_popt = []
    if lower_limit is None:
        lower_limit = np.nonzero(x)[0][0]
    # begin iteration
    for i in range(maxiters):
        # truncate data
        s = slice(lower_limit, upper_limit)
        yf, xf = y[s], x[s]
        # if the first iteration estimate
        if i < 1:
            if include_offset:
                # first find best fit for power law
                popt_no_offset, ul = fit_power_law(x, y, maxiters=maxiters,
                                                   floor=floor, upper_limit=upper_limit,
                                                   lower_limit=lower_limit, include_offset=False)
                # then estimate offset
                popt = np.append(popt_no_offset, y[y > 0].mean())
            else:
                popt = estimate_power_law(xf, yf)
        # fit truncated curve
        popt, pcov = curve_fit(power_law, xf, yf, p0=popt, jac=power_law_jac, method="mle")
        # append to list
        all_popt.append(popt)
        # check to see if we've converged
        if any(np.allclose(old_popt, popt) for old_popt in all_popt[:-1]):
            break
        # if we're not looking for an offset update truncation factor
        if not include_offset:
            upper_limit = int(power_intercept(popt, floor))
        else:
            if popt[-1] < 0:
                warnings.warn("offset less than 0, replacing with no offset params")
                popt = popt_no_offset
                break
            upper_limit = int(power_intercept(popt[:2], popt[-1]))
    else:
        if maxiters > 1:
            warnings.warn("Max iters reached")
    return popt, upper_limit


def fit_and_plot_power_law(trace, ul, offset, xmax, ll=None, ax=None, density=False, norm=False):
    """"""
    # calculate histogram (get rid of 0 bin)
    y = np.bincount(trace)[1:]
    x = np.arange(len(y) + 1)[1:]

    if ax is None:
        fig, ax = plt.subplots(1)
    popt, ul = fit_power_law(x, y, 10, include_offset=offset, upper_limit=ul, lower_limit=ll)

    # build display equation
    eqn = r"$\alpha = {:.2f}"
    if len(popt) > 2:
        eqn += ", offset = {:.2f}"
    eqn += "$"

    # build fit
    ty = power_law(x, *popt)
    ymin = 0.5
    if density or norm:
        if density:
            N = len(trace)
        elif norm:
            N = y[0]
        else:
            raise RuntimeError("Shouldn't be here")
        y = y / N
        ty /= N
        ymin /= N

    # plot on loglog plot
    ax.loglog(x, y, ".", label="Data")
    ax.loglog(x, ty, label=eqn.format(*popt[1:]))
    ax.set_ylim(bottom=ymin)

    if ll:
        ax.axvline(ll, color="y", linewidth=4, alpha=0.5, label="$x_{{min}} = {}$".format(ll))
    if ul:
        ax.axvline(ul, color="y", linewidth=4, alpha=0.5, label="$x_{{max}} = {}$".format(ul))

    # labeling
    if density:
        ax.set_ylabel("Frequency")
    elif norm:
        ax.set_ylabel("Fraction of Maximum")
    else:
        ax.set_ylabel("Occurences (#)")
    ax.set_xlabel("Number of frames")

    return plt.gcf(), ax, popt, ul


def stretched_exp(tdata, tau, beta):
    """A streched exponential distribution"""
    mean_tau = tau * gamma(1 + 1 / beta)
    return np.exp(-(tdata / tau) ** beta) / mean_tau


def power_law_cont(tdata, alpha, tmin=1):
    """A continuous power law distribution"""
    return ((alpha - 1) / tmin)(tdata / tmin) ** (-alpha)


def stretched_exp_ccdf(tdata, tau, beta):
    """Closed form cCDF for a stretched exponential distribution"""
    return gammaincc(1.0 / beta, (tdata / tau) ** beta) / (beta * gamma(1. + 1. / beta))


def power_law_cont_ccdf(tdata, alpha, tmin=1):
    """Closed form cCDF for a power law distribution"""
    # below is for discrete variables
    # return zeta(alpha, tdata) / zeta(alpha, tmin)
    return (tdata / tmin) ** (1 - alpha)


def se_pl_ccdf(tdata, tau, beta, alpha, f, pre, tmin=1):
    """the cCDF of a mixture of a stretched exponential and a power law distributions

    f is the fraction of stretched exponential in the total distribution."""
    ccdf_se = stretched_exp_ccdf(tdata, tau, beta)
    ccdf_pl = power_law_cont_ccdf(tdata, alpha, tmin)
    ccdf = ccdf_se * f + ccdf_pl * (1 - f)
    return ccdf * pre


def fit_and_plot_se_pl_ccdf(offtimes, density=True, ax=None):
    """Personal communication from Muzhou Wang (Northwestern)

    He fits the complementary cumulative distribution of offtimes to a mixture of
    a stretched exponential and a power law

    This function does the fit and makes the plot."""

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # make the distribution of off times
    y = np.bincount(offtimes)
    # truncate the 0 bin (no data)
    y = y[1:]
    x = np.arange(len(y)) + 1

    # make the experimental cumulative distribution function.
    ccdf = y[::-1].cumsum()[::-1]
    if density:
        ccdf = ccdf / ccdf[0]

    # make some reasonable starting guesses for the parameters
    p0 = np.percentile(offtimes, 95), 1, 2, 0.5, ccdf[0]
    try:
        popt, pcov = curve_fit(se_pl_ccdf, x, ccdf, p0=p0,
                               bounds=((0, 0, 0, 0, 0), (np.inf, np.inf, np.inf, 1, np.inf)))
    except RuntimeError:
        logger.warning("fit failed with starting variables = {}".format(p0))
        popt = None

    ax.loglog(x, ccdf, "C0.", label="Data")

    # if fit is successful plot it, its components and summary in the legend
    if popt is not None:
        tau, beta, alpha, f, pre = popt
        ax.loglog(x, se_pl_ccdf(x, *popt), "C1", label="Fit")

        ccdf_se = stretched_exp_ccdf(x, tau, beta)
        ccdf_pl = power_law_cont_ccdf(x, alpha, tmin=1)

        ax.loglog(x, ccdf_pl * (1 - f) * pre, "C2--", label="{:.0%} Power Law: $\\alpha = {:.2f}$".format(1 - f, alpha))
        ax.loglog(x, ccdf_se * f * pre, "C2:", label="{:.0%} Str. Exp.: $\\tau = {:}$, $\\beta = {:.2f}$".format(f, latex_format_e(tau), beta))

        ax.legend()
    ax.set_xlabel("# of Frames")
    ax.set_ylabel("cCDF")
    return fig, ax


def plot_occurences(samples_blinks, num=10, ax=None):
    """Make a plot of occurences as a function of number of frames.alpha

    The 1 curve shows the fraction of molecules that have occured *only* once at that frame

    the 2 cuve shows the fraction of molecules that have occured *only* twice by that frame.alpha

    etc."""

    # make axes if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # find on frames
    all_onframes = pd.concat([calc_onframes(s) for s in tqdm.tqdm(samples_blinks, desc="Getting on frames")], ignore_index=True)
    # calculate the number of occurences in each frame
    frames_and_occurrences = all_onframes.groupby(["occurrence", "frame"]).size()

    # reset the index to start at 1 for loglog plottings
    frames_and_occurrences.index = frames_and_occurrences.index.set_levels(frames_and_occurrences.index.levels[1] - frames_and_occurrences.index.levels[1].min() + 1, level=1)
    frames_and_occurrences /= frames_and_occurrences[1].sum()

    for i in range(1, num + 1):
        trace = frames_and_occurrences[i].sub(frames_and_occurrences[i + 1], fill_value=0).cumsum()
        trace.plot(ax=ax, label=str(i))

    ax.legend()
    ax.set_ylabel("Fraction of Molecules Observed # of Times")

    all_blinks = pd.concat(samples_blinks, ignore_index=True)
    start_frame, end_frame = all_blinks.frame.min(), all_blinks.frame.max()

    if start_frame > 1:
        ax.set_xlabel("Frame # - ${}$".format(latex_format_e(start_frame)))
    else:
        ax.set_xlabel("Frame #")

    return fig, ax


def plot_blinks(gap, max_frame, onofftimes=None, samples_blinks=None, ax=None, min_sigma=0, coords="xyz", density=False):
    """Plot blinking events given on off times and fitted power law decay of off times"""
    if ax is None:
        fig, ax = plt.subplots(1)

    # calculate the number of events for each purported molecule
    if samples_blinks is not None:
        logger.debug("Correcting blinks")
        samples_blinks = dask.compute(samples_blinks)[0]

        grouped = [dask.delayed(fast_group)(s, gap) for s in samples_blinks]
        aggs = [dask.delayed(agg_groups)(g) for g in grouped]

        regroup = dask.delayed([dask.delayed(cluster_groups)(agg, min_sigma, coords=coords) for agg in aggs])
        blinks = np.concatenate(regroup.compute(scheduler="processes"))
        prefix = "Corrected\n"
    elif onofftimes is not None:
        logger.debug("Not correcting blinks")
        blinks = np.array([count_blinks(s, gap) for s in onofftimes])
        prefix = "Uncorrected\n"
    else:
        raise RuntimeError

    blinks = blinks.astype(int)

    # Fit to zero-truncated poisson
    for guess in itt.product(*[(0.1, 0.5, 1, 2, 3)]*2):
        try:
            alpha, mu = fit_ztnb(blinks, x0=guess)
            # if successful fit, break
            break
        except RuntimeError as e:
            err = e
            continue
    else:
        logger.warn(err)
        alpha = mu = None
    # calculate histograms
    x = np.arange(blinks.max() + 1)[1:]
    y = np.bincount(blinks)[1:]
    label = "$\mu_{{data}}={:.2f}$, $p_{{50^{{th}}}}={}$".format(blinks.mean(), int(np.median(blinks)))
    # normalize
    N = y.sum()
    y = y / N

    # plot fit
    if alpha:
        fit = NegBinom(alpha, mu).pmf(x) / (1 - NegBinom(alpha, mu).pmf(0))
        label += "\n" + r"$\alpha={:.2f}$, $\mu_{{\lambda}}={:.2f}$, $\sigma_{{\lambda}}={:.2f}$".format(alpha, mu, mu / np.sqrt(alpha))
        if density:
            ax.set_ylabel("Frequency")
        else:
            ax.set_ylabel("Occurences (#)")
            fit = fit * N
            y = y * N

        ax.step(x, fit, where="mid", color="k", linestyle=":")

    label += "\nGap = {:g}, % Trace = {:.1%}".format(gap, gap / max_frame)

    ax.step(x, y, label=label, where="mid")
    ax.legend()
    ax.set_xlim(left=0, right=30)
    ax.set_xlabel("# of events / molecule")
    ax.set_title("Long Term Blinking")

    return plt.gcf(), ax, blinks


def prune_blobs(df, radius):
        """
        Pruner method takes blobs list with the third column replaced by
        intensity instead of sigma and then removes the less intense blob
        if its within diameter of a more intense blob.

        Adapted from _prune_blobs in skimage.feature.blob

        Parameters
        ----------
        blobs : ndarray
            A 2d array with each row representing 3 values,
            `(y, x, intensity)` where `(y, x)` are coordinates
            of the blob and `intensity` is the intensity of the
            blob (value at (x, y))
        diameter : float
            Allowed spacing between blobs

        Returns
        -------
        A : ndarray
            `array` with overlapping blobs removed.
        """

        # make a copy of blobs otherwise it will be changed
        # create the tree
        blobs = df[["x0", "y0", "amp"]].values
        kdtree = cKDTree(blobs[:, :2])
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
            key=lambda x: max(blobs[x[0], -1], blobs[x[1], -1]),
            reverse=True
        )
        # indices of pruned blobs
        pruned_blobs = set()
        # loop through conflicts
        for idx_a, idx_b in tqdm.tqdm(list_of_conflicts, desc="Removing conflicts"):
            # see if we've already pruned one of the pair
            if (idx_a not in pruned_blobs) and (idx_b not in pruned_blobs):
                # compare based on amplitude
                if blobs[idx_a, -1] > blobs[idx_b, -1]:
                    pruned_blobs.add(idx_b)
                else:
                    pruned_blobs.add(idx_a)
        # generate the pruned list
        # pruned_blobs_set = {(blobs[i, 0], blobs[i, 1])
        #                         for i in pruned_blobs}
        # set internal blobs array to blobs_array[blobs_array[:, 2] > 0]
        return df.iloc[[
            i for i in range(len(blobs)) if i not in pruned_blobs
        ]]


def check_density(shape, data, mag, thresh_func=thresholding.threshold_triangle, diagnostics=True, dim=3, zscaling=5 * 130):
    """Check the density of the data and find the 99th percentile density

    Return a function that calculates maximal grouping gap from grouping radius

    Allow 2 and 3 dimensional calculations"""

    # generate the initial histograms of the data
    if dim == 3:
        hist3d = save_img_3d(shape, data, None, zspacing=zscaling / mag, mag=mag, hist=True)
        hist2d = montage(hist3d)
    elif dim == 2:
        hist2d = gen_img(shape, data, mag=mag, hist=True, cmap=None).astype(int)
    else:
        raise RuntimeError("dim of {} not allowe, must be 2 or 3".format(dim))

    # convert hist2d to density (# molecules per pixel or voxel)
    hist2d = hist2d * (mag ** dim)
    ny, nx = hist2d.shape

    min_density = 0

    # remove places with low values
    to_threshold = hist2d[hist2d > min_density]
    to_threshold = to_threshold[to_threshold < np.percentile(hist2d[hist2d > min_density], 99)]

    thresh = thresh_func(to_threshold)
    max_thresh = np.percentile(hist2d[hist2d > thresh], 99)

    # calculate density in # molecules / unit area (volume) / frame
    frame_range = data.frame.max() - data.frame.min()
    frame_density = max_thresh / frame_range

    if diagnostics:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8 * nx / ny, 4))

        ax0.matshow(hist2d, norm=PowerNorm(0.5), vmax=max_thresh, cmap="Greys_r")
        ax0.set_title("Data")
        ax1.matshow(hist2d, norm=PowerNorm(0.5), vmin=thresh, vmax=max_thresh, cmap=greys_limit)
        ax0.set_title("Data and Thresholds")

        # plot of histogram
        fig3, ax0 = plt.subplots(1)
        ax0.hist(hist2d.ravel(), bins=np.logspace(1, np.log10(hist2d.max() + 10), 128), log=True, color="k")
        ax0.set_xscale("log")
        ax0.axvline(max_thresh, c="r", label="99% Thresh = {:g} # / area\n= {:g} # / area / frame".format(max_thresh, frame_density))
        ax0.axvline(thresh, c="b", label="{} = {:g} # / area".format(thresh_func.__name__, thresh))
        ax0.legend()
        ax0.set_title("Histogram of Histogram Values")

        # to_plot = hist2d[hist2d > 1]
        # to_plot = to_plot[to_plot < max_thresh]
        # ax1.hist(to_plot, bins=128, log=True, color="k")
        fig3.tight_layout()

    def gap(r):
        if dim == 3:
            return 1 / (4 / 3 * r ** 3 * frame_density)
        return 1 / (r ** 2 * frame_density)

    return gap, frame_density


def mortensen(df, a2=1.0):
    """Calculate the mortensen precision of the emitter

    https://www.nature.com/articles/nmeth.1447 (https://doi.org/10.1038/nmeth.1447)

    a2 is the area of the pixel, as the widths are in pixels this should be 1"""
    var_n = (df[["width_x", "width_y"]] ** 2).div(df.nphotons, "index")
    # b^2 in the paper is the background photon count _not_ the background photon count squared ...
    b2 = df.offset
    new_var = var_n * (16 / 9 + 8 * np.pi * var_n.mul(b2, "index") / a2)
    new_std = np.sqrt(new_var)
    new_std.columns = "mort_x", "mort_y"
    return pd.concat((df, new_std), axis=1)


def bin_by_photons(blob, nphotons):
    # order by z, so that we group things that are near each other in frames (even if they're in different slabs)
    # this also, conveniently, makes a copy of the DF
    blob = blob.sort_values("frame")

    # Calculate the total number of photons and then the bin edges
    # such that the bins each have n photons in them
    total_nphotons = blob.nphotons.sum()
    bins = np.arange(0, total_nphotons, nphotons)

    # break the DF into groups with n photons
    blob2 = blob.assign(group_id=blob.groupby(pd.cut(blob.nphotons.cumsum(), bins)).grouper.group_info[0])
    # group and calculate the mortensen precision
    # we recalc mortensen because the new agg has an average width and sum number of photons
    blob2 = blob2.drop(["mort_x", "mort_y"], axis=1, errors="ignore")
    return mortensen(agg_groups(blob2))


def scale_func(ydata, scale, bg):
    """scale xdata to ydata"""
    return scale * ydata + bg


def mort(xdata, scale, bg):
    """scale xdata to ydata"""
    return (xdata - bg) / scale


def calc_precision(blob, nphotons=None):
    # set up data structures
    # experimental precision
    expt = []
    # mortensen precision
    mort = []

    #
    if nphotons is None:
        # total number of photons
        total_photons = blob.nphotons.sum()
        # at the end we want a single group
        # nphotons = np.logspace(np.log10(blob.nphotons.median() * 2), np.log10(total_photons / 3), max(3, int(np.sqrt(len(blob)))))
        nphotons = np.logspace(3.5, 5, 32)
        logger.debug("nphotons is {}".format(nphotons))

    for n in nphotons:
        binned_blob = bin_by_photons(blob, n)
        expt.append(binned_blob[["x0", "y0", "z0"]].std())
        mort.append(binned_blob[["sigma_x", "sigma_y", "sigma_z", "mort_x", "mort_y"]].mean())

        # expt.append(blob3[["x0", "y0"]].std())
        # mort.append(blob3[["mort_x", "mort_y"]].mean())

    expt_df = pd.DataFrame(expt, index=nphotons)
    mort_df = pd.DataFrame(mort, index=nphotons)
    precision_df = pd.concat((expt_df, mort_df), 1).dropna()
    return precision_df


def nnlr(xdata, ydata):
    """Non-negative linear regression, a linear fit where both b and m are greater than zero"""
    lhs = np.stack((xdata, np.ones_like(xdata)), -1)
    rhs = ydata
    (m, b), resid = nnls(lhs, rhs)
    return m, b


def fit_precision(precision_df, non_negative=False, diagnostics=False):
    fits = {}
    fit_df = []
    for c in "xyz":
        types = "mort_", "sigma_"
        if c == "z":
            types = types[1:]
        for t in types:
            expt_df, calc_df = precision_df[c + "0"], precision_df[t + c]

            if non_negative:
                # non-negative fitting will tend to dump bad fits
                # into 0 offset 0 scale, not really what we want.
                popt = nnlr(calc_df, expt_df)
            else:
                popt = np.polyfit(calc_df, expt_df, 1)

            fits[t + c + "_scale"] = popt[0]
            fits[t + c + "_offset"] = popt[1]
            fit_df.append(scale_func(calc_df, *popt))

    if diagnostics:
        fit_df = pd.concat(fit_df, axis=1)
        fig, ((ax_z, ax_fit_z), (ax_xy, ax_fit_xy)) = plt.subplots(2, 2, sharex=True, sharey="row", figsize=(9, 9))

        # turn into nm
        precision_df = precision_df * 130
        precision_df[["z0", "sigma_z"]] /= 130
        fit_df = fit_df * 130
        fit_df[["sigma_z"]] /= 130

        precision_df[["z0", "sigma_z"]].plot(ax=ax_z, style=":o")

        precision_df[["z0"]].plot(ax=ax_fit_z, style=":o")
        fit_df[["sigma_z"]].plot(ax=ax_fit_z, style=":o")

        precision_df[["x0", "y0"]].plot(ax=ax_xy, style=":o")
        precision_df[["mort_x", "mort_y"]].plot(ax=ax_xy, style=":o")
        precision_df[["sigma_x", "sigma_y"]].plot(ax=ax_xy, style=":o")

        precision_df[["x0", "y0"]].plot(ax=ax_fit_xy, style=":o")
        fit_df[["mort_x", "mort_y", "sigma_x", "sigma_y"]].plot(ax=ax_fit_xy, style=":o")

        ax_z.set_title("Data")
        ax_fit_z.set_title("Fits")

        ax_z.set_ylabel("Group Precision (nm)")
        ax_fit_xy.set_ylabel("Group Precision (pixels)")

        ax_fit_xy.set_xlabel("~# of Photons")
        ax_xy.set_xlabel("~# of Photons")
        ax_xy.set_ylabel("Group Precision (nm)")

        fig.tight_layout()
    return fits


def fix_title(t):
    try:
        kind, c, measure = t.split("_")
    except ValueError:
        return t.capitalize()
    if kind == "mort":
        kind = "Mortensen"
    else:
        kind = "Calculated"

    measure = measure.capitalize()
    return r"{} $\sigma_{}$ {}".format(kind, c, measure)


def plot_fit_df(fits_df, minscale=5e-1, minoffset=-1e-3):
    """"""
    # columns
    scale_col = ["mort_x_scale", "sigma_x_scale", "mort_y_scale", "sigma_y_scale"]
    scale_col_z = ["sigma_z_scale"]
    offset_col = ["mort_x_offset", "sigma_x_offset", "mort_y_offset", "sigma_y_offset"]
    offset_col_z = ["sigma_z_offset"]
    other_col = fits_df.columns.difference(scale_col + scale_col_z + offset_col + offset_col_z).tolist()

    # set up plot
    fig, axs = plt.subplots(4, 4, sharex="row", sharey="row", figsize=(16, 16))

    # filter data frame
    filt_fits_df = fits_df[(fits_df[scale_col + scale_col_z] > minscale).all(1)]
    filt_fits_df = filt_fits_df[(filt_fits_df[offset_col + offset_col_z] > minoffset).all(1)]

    # scale xy offsets to nm
    filt_fits_df[offset_col] *= 130

    # make hists
    filt_fits_df[scale_col].hist(bins=np.linspace(0, 3, 32), density=True, ax=axs[0])
    filt_fits_df[offset_col].hist(bins=np.linspace(0, 35, 32), density=True, ax=axs[1])

    filt_fits_df[scale_col_z].hist(bins=np.linspace(0, 3, 32), density=True, ax=axs[2, 0])
    filt_fits_df[offset_col_z].hist(bins=np.linspace(0, 350, 32), density=True, ax=axs[3, 0])

    # clean up titles
    for ax in axs.ravel():
        ax.set_title(fix_title(ax.get_title()))

    # clean up grid
    dplt.clean_grid(fig, axs)
    fig.tight_layout()

    # add tables of summary statistics
    tab_ax0 = fig.add_axes((0.25, 0.25, 0.75, 0.25))
    tab_ax1 = fig.add_axes((0.25, 0, 0.75, 0.25))

    for tab_ax, col in zip((tab_ax0, tab_ax1), (scale_col + scale_col_z, offset_col + offset_col_z + other_col)):
        tab_ax.axis("off")
        dcsummary = filt_fits_df.describe().round(3)[col]
        tab = tab_ax.table(cellText=dcsummary.values, colWidths=[0.15] * len(dcsummary.columns),
                           rowLabels=dcsummary.index,
                           colLabels=[fix_title(c) for c in dcsummary.columns],
                           cellLoc='center', rowLoc='center',
                           loc='center')
        tab.scale(0.9, 2)
        tab.set_fontsize(16)

    axs_scatter = pd.plotting.scatter_matrix(filt_fits_df[fits_df.columns.difference(other_col)], figsize=(20, 20))

    for ax in axs_scatter.ravel():
        ax.set_ylabel(fix_title(ax.get_ylabel()))
        ax.set_xlabel(fix_title(ax.get_xlabel()))

    return (fig, axs), (ax.get_figure(), axs_scatter)


def cluster(data, features=["x0", "y0"], scale=True, diagnostics=False, algorithm=HDBSCAN, copy_data=False, **kwargs):
    """
    Cluster PALM data based on given features
    
    Parameters
    ----------
    data : pd.DataFrame
        DF holding PALM data
    features : list-like
        Features of data to use for clustering
    scale : bool (default True)
        Prescale the data using sklearn.preprocessing.StandardScaler
    algorithm : sklearn like cluster algorithm (default HDBSCAN)
        Algorithm used to cluster the data, must provide `fit` method and `labels_` attribute
    diagnostics : bool (default True)
        show diagnostic plots (don't use for large data frames!)
    copy_data : bool (default False)
        make a copy of the data, if False a column called group_id will be added to input DF

    All extra keyword arguments will be passed to `algorithm`
    """
    X = data[features]
    if scale:
        X = StandardScaler().fit_transform(X)

    cluster = algorithm(**kwargs).fit(X)

    if diagnostics:
        with plt.style.context("dark_background"):
            fig, (ax0, ax1, ax2) = plt.subplots(3, figsize=(3, 9))
            data.plot("x0", "y0", c="frame", edgecolor="w", linewidth=0.5, kind="scatter", cmap="gist_rainbow", ax=ax0)
            ax1.scatter(data.x0, data.y0, c=cluster.labels_, cmap="gist_rainbow")
            ax2.bar(np.unique(cluster.labels_), np.bincount(cluster.labels_ + 1))

    if copy_data:
        data = data.copy()

    data["group_id"] = cluster.labels_
    return data


def fit_plot_offtimes(offtimes, ax=None):
    """Utility function for fitting and plotting"""
    fig, ax, popt, ul = fit_and_plot_power_law(offtimes, None, None, None, ax=ax, norm=True)
    # remove xmin lines
    ax.get_lines().pop().remove()
    ax.set_title("Off Times")
    return fig, ax, popt


def fit_plot_ontimes(ontimes, ax=None):
    """Utility function for fitting and plotting"""
    power = PowerLaw(ontimes)
    power.fit(xmin_max=20)
    fig, ax = power.plot(ax, norm=True)
    # remove xmin lines
    ax.get_lines().pop().remove()
    ax.set_title("On Times")
    return fig, ax, power


def cutoffs(offtimes, percentiles):
    """Find out when the curve falls to percent of max"""
    percentiles = np.asarray(percentiles)

    # turn into histogram
    y = np.bincount(offtimes)[1:]
    x = np.arange(offtimes.max()) + 1

    # get conjugate percentiles
    cp = 1 - percentiles

    # norm y
    ynorm = (y / y[0])

    # find percentiles that will work with our data
    valid_cps = ((ynorm[:, None] < cp) & (ynorm[:, None] > np.zeros_like(cp))).any(0)
    cp = cp[valid_cps]

    # search for these in the hist
    search = np.abs(ynorm[:, None] - cp[None])
    minval = search.min(0)

    # take median of any valid values
    cuts = np.array([int(np.median(x[filt])) for filt in (search == minval).T])
    return cuts, cp


def make_extract_fiducials_func(data):
    """Returns a function for extracting fiducials from data"""
    # build and enclose tree for faster compute
    tree = cKDTree(data[["y0", "x0"]].values)

    @dask.delayed
    def grab(m):
        return data.iloc[m]

    def extract_fiducials(blobs, radius, minsize=0, maxsize=np.inf):
        matches = tree.query_ball_point(blobs, radius)
        new_matches = filter(lambda m: minsize < len(m) < maxsize, matches)

        return [grab(m) for m in new_matches]

    return extract_fiducials


def get_onofftimes(samples, extract_radius, data=None, extract_fiducials=None, prune_radius=2):
    """Calculate onofftimes from samples"""
    if extract_fiducials is None:
        if data is None:
            raise ValueError("Must pass data or extract_fiducials function")
        extract_fiducials = make_extract_fiducials_func(data)

    samples = prune_blobs(samples, prune_radius)

    # extract samples
    n = len(samples)
    logging.info("Extracting {} samples ...".format(n))
    samples_blinks = extract_fiducials(samples[["y0", "x0"]].values, extract_radius)
    logging.info("Kept samples = {}%".format(int(len(samples_blinks) / n * 100)))

    # calculate on/off times
    logging.info("Calculating on and off times ... ")
    onofftimes = dask.delayed([dask.delayed(on_off_times_fast)(y) for y in samples_blinks]).compute()

    return onofftimes, samples_blinks


def plot_onofftimes(onofftimes, max_frame, axs=None):
    """Plot on and off times"""
    # now compute on and offtimes
    ontimes, offtimes = get_on_and_off_times(onofftimes)

    # make figure if none is provided
    if axs is None:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    else:
        fig = axs.ravel()[0].get_figure()

    # add plots
    _, _, power = fit_plot_ontimes(ontimes, ax=axs[0, 0])
    _, _, popt = fit_plot_offtimes(offtimes, ax=axs[0, 1])

    # add legends
    for ax in axs[0]:
        ax.legend()

    # add cutoffs
    # 1e-2, 1e-4, 1e-6
    my_cutoffs = cutoffs(offtimes, (0.99, 0.999, 0.9999))
    for gap, cp in zip(*my_cutoffs):
        line = mlines.Line2D([gap, gap, 0], [0, cp, cp], color="r")
        axs[0, 1].add_line(line)
        plot_blinks(gap, max_frame, onofftimes, density=True, ax=axs[1, 1])

    # calculate ratios
    _, ratio = calc_onoff_ratio(onofftimes)

    # add ratios to the plot with legend
    ax = axs[1, 0]
    hist_and_cumulative(ratio, ax=ax, log=True)
    ax.set_xlabel("Ratio of On Time to Off Time")
    ax.set_title("Dynamic Contrast Ratio")
    median = np.median(ratio)
    ax.axvline(median, ls=":", color="k", label="Median = {:.2g}".format(median))
    ax.legend(loc='lower right')

    return fig, axs


def plot_onofftimes2(onofftimes, samples_blinks, xlims=(1e3, 1e6)):
    """Make a more detailed on off times plot"""

    all_blinks = pd.concat(samples_blinks, ignore_index=True)
    start_frame, end_frame = all_blinks.frame.min(), all_blinks.frame.max()

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    plot_onofftimes(onofftimes, end_frame, axs)

    # split on and off times
    ontimes, offtimes = get_on_and_off_times(onofftimes, True)

    # add cumulative distribution
    fit_and_plot_se_pl_ccdf(offtimes, ax=axs[0, -1])

    # add occurences
    ax_occur = axs[-1, -1]
    plot_occurences(samples_blinks, 7, ax_occur)
    ax_occur.set_xscale("log")
    ax_occur.set_xlim(*xlims)

    return fig, axs


def do_trace_analysis(palm_data, basetitle, samples, directory="./", fractions=np.logspace(-1, 0, 3), radii=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """Make and save trace analyses"""
    extract_fiducials = make_extract_fiducials_func(palm_data)

    # iterate over sampling fractions
    for f in fractions:
        # include samples by closure, these won't change
        new_samples = samples.sample(frac=f)

        # iterate over radii
        for radius in radii:
            # extract samples
            onofftimes, samples_blinks = get_onofftimes(new_samples, radius, extract_fiducials=extract_fiducials, prune_radius=1)
            # if basetitle isn't formated correctly then we want to fail here
            t = basetitle.format(radius, len(samples_blinks))
            # process samples
            samples_blinks = dask.delayed(samples_blinks).compute()
            # make the figure
            fig, axs = plot_onofftimes2(onofftimes, samples_blinks)
            # save the figure
            fig.suptitle(t, y=1.01)
            fig.tight_layout()
            fig.savefig(directory + t + ".png", dpi=300, bbox_inches="tight")


def render_and_save(palm, directory):
    """Utility function to render and save PALM images"""
    for sxy in (0.1, 0.25):
        for zp in (0.2, ):
            for mag in (10, 25):
                to_render = palm.grouped_nf[(palm.grouped_nf[["sigma_x", "sigma_y"]] < sxy).all(1)]
                zmin, zmax = to_render.z0.quantile((zp, 1 - zp))
                to_render = to_render[(zmin < to_render.z0) & (to_render.z0 < zmax)]

                cimg = gen_img(palm.shape, to_render, mag=mag, diffraction_limit=True, zscaling=5 * 130)
                cimg.zrange = np.array(cimg.zrange) / 1000
                fig, ax = cimg.plot(norm_kwargs=dict(auto=True), subplots_kwargs=dict(figsize=(10, 8)))

                fname = directory + "{} " + "{}X Grouped PALM Dim={}, r={:.2f}, gap={}, sxy lt {:.2f}".format(cimg.mag, 3, palm.group_radius, palm.group_gap, sxy) + "{}"
                cimg.save_color(fname.format("Depthcoded", ".png"), auto=True)
                cimg.save_alpha(fname.format("Alpha", ".tif"))


def get_on_and_off_times(onofftimes, clip=False):
    """Get on times and off times from onofftimes"""
    if clip:
        s = slice(None, -1)
    else:
        s = slice(None)
    ontimes = np.concatenate([b[0][s] for b in onofftimes]).astype(int)
    offtimes = np.concatenate([b[1] for b in onofftimes]).astype(int)
    # p = stats.pearsonr(offtimes, ontimes)
    # s = stats.spearmanr(offtimes, ontimes)

    # print("Pearson correlation {:.2g}, p-value {:.2g}".format(*p))
    # print("Spearman correlation {:.2g}, p-value {:.2g}".format(*s))
    return ontimes, offtimes


def calc_onoff_ratio(onofftimes):
    ontimes, offtimes = get_on_and_off_times(onofftimes, clip=True)
    # ratio of on time to following off time
    ratio = ontimes / offtimes
    # ratio2 is the total ontime for a molecule divided by total on time
    ratio2 = np.array([b[0][:-1].sum() / b[1].sum() for b in onofftimes if b[1].sum() > 0])
    ratio2 = ratio2[np.isfinite(ratio2)]
    return ratio, ratio2


def hist_and_cumulative(data, ax=None, log=False):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if log:
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 64)
    else:
        bins = np.linspace(data.min(), data.max(), 64)
    ax.hist(data, bins=bins, density=True, log=False, histtype="step")
    ax.set_ylabel("PDF")
    if log:
        ax.set_xscale("log")

    sorted_data = np.sort(data)
    N = len(sorted_data)
    b = np.arange(N) / N

    twin_ax = ax.twinx()
    color = ax._get_lines.get_next_color()
    twin_ax.plot(sorted_data, b, color=color, ls="steps-mid")
    twin_ax.tick_params(axis='y', labelcolor=color)
    twin_ax.set_ylabel("CDF", color=color)
    twin_ax.set_ylim(bottom=0)

    return fig, ax


def calc_onframes(df):
    """Pull frames where molecule turns on"""
    # make a trace
    trace = np.sort(df.frame.unique())

    # calculate the spacing between events
    diff = np.append(1, np.diff(trace))

    # frames where this molecule turns on
    on_idx = diff > 1

    # first frame should be true
    on_idx[0] = True
    on_frames = trace[on_idx]

    return pd.DataFrame(data=np.stack((on_frames, np.arange(len(on_frames)) + 1)).T, columns=["frame", "occurrence"])

"""
# build tree for use later
extract_fiducials = make_extract_fiducials_func(data)

n = len(samples)

offtimes_list = []

for f in np.logspace(-1,0,3):
    samples = data_filt.groupby("group_id").mean().sample(frac=f)
    samples = pdiag.prune_blobs(samples, 2)
    n = len(samples)
    
    for radius in (0.1, 0.2, 0.3, 0.4, 0.5):
        gc.collect()
        # extract samples
        print("Extracting {} samples ...".format(n))
        %time samples_blinks = extract_fiducials(data, samples[["y0", "x0"]].values, radius)
        print("Kept samples = {}%".format(int(len(samples_blinks) / n * 100)))

        # calculate localizations per sample
        # make generator, memory efficient ...
        # calculate on/off times
        print("Calculating on and off times ... ")
        onofftimes = dask.delayed([dask.delayed(pdiag.on_off_times_fast)(y) for y in samples_blinks])
        with pdiag.pb:
            onofftimes = onofftimes.compute()
        ontimes = np.concatenate([b[0] for b in onofftimes]).astype(int)
        offtimes = np.concatenate([b[1] for b in onofftimes]).astype(int)
        
        offtimes_list.append(offtimes)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        fit_plot_ontimes(ontimes, ax=axs[0])
        _, _, popt = fit_plot_offtimes(offtimes, ax=axs[1])

        for ax in axs[:2]:
            ax.legend()
        # 1e-2, 1e-4, 1e-6
        for gap, cp in zip(*cutoffs(offtimes, (0.99, 0.999, 0.9999, 0.999999))):
            line = mlines.Line2D([gap, gap, 0], [0, cp, cp], color="r")
            ax.add_line(line)
            pdiag.plot_blinks(gap, max_frame, onofftimes, density=True, ax=axs[2])
        # with pdiag.pb:
        #     pdiag.plot_blinks(gap, max_frame, samples_blinks=samples_blinks, ax=axs[2])

        t = "Halo-TOMM20 (JF525) 4K Blinking, 20170519, radius = {:.2f}, Sample size = {}".format(radius, len(samples_blinks))
        fig.suptitle(t, y=1.01)
        fig.tight_layout()
        
        fig.savefig(t + ".png", dpi=300, bbox_inches="tight")
"""