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
from matplotlib.colors import PowerNorm, ListedColormap

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

# override any earlier imports
from peaks.lm import curve_fit
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
        return tif.imread(path)


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

        # generate an index
        self.date_idx = pd.DatetimeIndex(
            data=np.concatenate(times),
            name="Timestamp"
        )

        assert len(self.date_idx) == len(self.raw), "Date index and Raw data don't match, {}, {}".format(self.date_idx, self.raw)

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
        self.shape = shape
        self.processed = processed
        self.drift = drift
        self.drift_corrected = drift_corrected
        self.grouped = grouped

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
        raw_df = peakselector_df(path_to_sav, verbose=verbose)
        # the dummy attribute won't stick around after casting, so pull it now.
        totalrawdata = raw_df.totalrawdata
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

        return cls(totalrawdata.shape, processed=processed, grouped=grouped)

    def drift_correct(self, sz=25, **kwargs):
        _, self.drift, _, self.drift_fiducials = remove_all_drift(
            self.processed[self.processed.sigma_z < sz],
            self.shape, None, None,
            **kwargs
        )
        self.drift_corrected = remove_drift(self.processed, self.drift)
        calc_fiducial_stats(self.drift_fiducials, diagnostics=True)

    def group(self, r, zscaling=10):
        """group the drift corrected data"""
        gap, thresh = check_density(self.shape, self.drift_corrected, 1, dim=3, zscaling=zscaling)
        mygap = int(gap(r))
        logger.info("Grouping gap is being set to {}".format(mygap))
        self.grouped = slab_grouper([self.drift_corrected], radius=r, gap=mygap, zscaling=zscaling * 130, numthreads=48)[0]

    @cached_property
    def fiducials(self):
        return find_fiducials(self.drift_corrected, self.shape, diagnostics=True)

    def remove_fiducials(self, radius):
        """remove fiducials from data"""
        for df_title in ("drift_corrected", "grouped"):
            try:
                df = getattr(self, df_title)
            except AttributeError:
                continue
            df_nf = filter_fiducials(df, self.fiducials, radius)
            setattr(self, df_title + "_nf", df_nf)

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

    def make_fiducial_mask(self, radius):
        """A mask at fiducial locations

        Note that this returns True for where the fiducials are and False
        For where they aren't"""
        mask = np.zeros(self.shape)
        for y, x in self.fiducials:
            mask[circle(y, x, radius, shape=self.shape)] = 1
        self.fiducial_mask = mask.astype(bool)


def exponent(xdata, amp, rate, offset):
    """Utility function to fit nonlinearly"""
    return amp * np.exp(rate * xdata) + offset


class Data405(object):
    """An object encapsulating function's related to reactivation data"""
    try:
        calibration = pd.read_excel("../Aggregated 405 Calibration.xlsx")
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

    def save(self, fname):
        """"""
        pd.Series(self.path).to_hdf(fname, "path405")

    @classmethod
    def load(cls, fname):
        """"""
        path = pd.read_hdf(fname, "path405").values[0]
        return cls(path)

    def fit(self, lower_limit, upper_limit=None):
        # we know that the laser is subthreshold below 0.45 V and the max is 5 V, so we want to limit the data between these two
        data_df = self.data
        if upper_limit is None:
            upper_limit = data_df.reactivation.max()
        data_df_crop = data_df[(data_df.reactivation > lower_limit) & (data_df.reactivation < upper_limit)].dropna()
        self.popt, self.pcov = curve_fit(exponent, *data_df_crop[["date_delta", "reactivation"]].values.T)
        data_df["fit"] = exponent(data_df["date_delta"], *self.popt)
        self.fit_win = data_df_crop.date_delta.min(), data_df_crop.date_delta.max()

    def plot(self, ax=None, limits=True, lower_limit=0.45, upper_limit=None):
        if ax is None:
            fig, ax = plt.subplots()
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
                        label = "fit limits"
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


def weird(xdata, *args):
    """Honestly this looks like saturation behaviour"""
    res = np.zeros_like(xdata)
    for a, b, c in zip(*(iter(args),) * 3):
        res += a * (1 + b * xdata) ** c
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

    def __init__(self, raw, palm, activation):
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

    @classmethod
    def load(cls, fname):
        """Load data from a pandas managed HDF5 store"""
        raw = RawImages.load(fname)
        palm = PALMData.load(fname)
        activation = Data405.load(fname)
        return cls(raw, palm, activation)

    def save(self, fname):
        """Save data to a pandas managed HDF store"""
        self.raw.save(fname)
        self.palm.save(fname)
        self.activation.save(fname)

    def masked_agg(self, masktype, agg_func=np.median):
        """Mask and aggregate raw data along frame direction with agg_func

        Save results on RawImages object"""
        # make sure our agg_func works with masked arrays
        agg_func = getattr(np.ma, agg_func.__name__)
        
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
            assert len(chunk) == len(chunked_shifts), "Lengths don't match"
            
            # convert image data to float
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
            return np.asarray(agg_func(shifted_masked_array, (1, 2)))

        # get x, y shifts from palm drift
        shifts = self.palm.drift[["y0", "x0"]].values

        # set up computation tree
        to_compute = [
            shift_and_mask(lazy_imread(path), shifts[cut_points[i]:cut_points[i + 1]])
            for i, path in enumerate(self.raw.paths)
        ]

        # get masked results
        masked_result = np.concatenate(dask.delayed(to_compute).compute(scheduler="processes"))
        
        # save these attributes on the RawImages Object
        attrname = "masked_" + masktype.lower() + "_" + agg_func.__name__
        logger.info("Setting {}".format(attrname))
        setattr(self.raw, attrname, masked_result)

        return masked_result

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
        setattr(self.raw, attrname, result)

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

    @cached_property
    def frame(self):
        """Make a groupby object that is by frame"""
        return self.palm.drift_corrected_nf.groupby("frame")

    @cached_property
    def nphotons(self):
        """mean number of photons per frame per localization excluding fiducials"""
        nphotons = self.frame.nphotons.mean().reindex(pd.RangeIndex(len(self.raw)))
        nphotons.index = self.time_idx
        return nphotons

    @cached_property
    def counts(self):
        """Number of localizations per frame, excluding fiducials"""
        counts = self.frame.size().reindex(pd.RangeIndex(len(self.raw)))
        counts.index = self.time_idx
        return counts

    @cached_property
    def intensity(self):
        """Median intensity within masked (data) area, in # of photons"""
        return pd.Series(data=self.raw.masked_data_median - 100, index=self.time_idx)

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

        # set figure title
        axs[0].set_title("Feedback Only")

        fig.tight_layout()
        
        return fig, axs

    def plot_all(self, **kwargs):
        """Plot entire experiment"""
        
        # make the figure
        fig, axs = plt.subplots(4, figsize=(6, 12), sharex=True)

        # Plot all data
        self._plot_sub(axs[:3])

        # add activation plot
        self.activation.plot(ax=axs[-1], limits=False, **kwargs)

        # add line for t = 0
        for ax in axs:
            ax.axvline(0, color="r")

        fig.tight_layout()
        
        return fig, axs

    def plot_nofeedback(self):
        """Plot many statistics for a PALM photophysics expt"""
        
        # end at 0 time
        fig, axs = self._plot_sub(None, s=slice(None, 0))

        axs[0].set_title("No Feedback")
        axs[-1].set_xlabel("Time (hours)")

        fig.tight_layout()
        return fig, axs

    def _plot_sub(self, axs=None, s=slice(None, None, None)):
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
            df = df.loc[s]
            df.plot(ax=ax)
            df.rolling(1000, 0, center=True).mean().plot(ax=ax)
            ax.set_ylabel(title)

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
    ax.loglog(x, y, linestyle="steps-mid", label="Data")
    ax.loglog(x, ty, label=eqn.format(*popt[1:]))
    ax.set_ylim(ymin=ymin)

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


def plot_blinks(gap, max_frame, onofftimes=None, samples_blinks=None, ax=None, min_sigma=0, coords="xyz", density=False):
    """Plot blinking events given on off times and fitted power law decay of off times"""
    if ax is None:
        fig, ax = plt.subplots(1)
        
    # calculate the number of events for each purported molecule
    if samples_blinks is not None:
        samples_blinks = dask.compute(samples_blinks)[0]

        grouped = [dask.delayed(fast_group)(s, gap) for s in samples_blinks]
        aggs = [dask.delayed(agg_groups)(g) for g in grouped]

        regroup = dask.delayed([dask.delayed(cluster_groups)(agg, min_sigma, coords=coords) for agg in aggs])
        blinks = np.concatenate(regroup.compute(scheduler="processes"))
        prefix = "Corrected\n"
    elif onofftimes is not None:
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
        raise err
    # calculate histograms
    x = np.arange(blinks.max() + 1)[1:]
    y = np.bincount(blinks)[1:]
    # normalize
    N = y.sum()
    y = y / N

    # plot fit
    fit = NegBinom(alpha, mu).pmf(x) / (1 - NegBinom(alpha, mu).pmf(0))

    label = "$\mu_{{data}}={:.2f}$, $p_{{50^{{th}}}}={}$\n".format(blinks.mean(), int(np.median(blinks)))
    label += r"$\alpha={:.2f}$, $\mu_{{\lambda}}={:.2f}$, $\sigma_{{\lambda}}={:.2f}$".format(alpha, mu, mu / np.sqrt(alpha))
    label += "\nGap = {:g}, % Trace = {:.1%}".format(gap, gap / max_frame)

    if density:
        ax.set_ylabel("Frequency")
    else:
        ax.set_ylabel("Occurences (#)")
        fit = fit * N
        y = y * N

    ax.step(x, y, label=label, where="mid")
    ax.step(x, fit, where="mid", color="k", linestyle=":")

    ax.legend()
    ax.set_xlim(xmin=0, xmax=30)
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
        for idx_a, idx_b in tqdm.tqdm(list_of_conflicts):
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


def check_density(shape, data, mag, thresh_func=thresholding.threshold_triangle, diagnostics=True, pix_size=130, dim=3, zscaling=5):
    """Check the density of the data and find the 99th percentile density

    Return a function that calculates maximal grouping gap from grouping radius

    Allow 2 and 3 dimensional calculations"""

    # generate the initial histograms of the data
    if dim == 3:
        hist3d = save_img_3d(shape, data, None, zspacing=pix_size * zscaling / mag, mag=mag, hist=True)
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


def make_extract_fiducials_func(data):
    # build and enclose tree for faster compute
    tree = cKDTree(data[["y0", "x0"]].values)

    def extract_fiducials(df, blobs, radius, minsize=0, maxsize=np.inf):
        matches = tree.query_ball_point(blobs, radius)
        new_matches = filter(lambda m: minsize < len(m) < maxsize, matches)

        @dask.delayed
        def grab(m):
            return df.iloc[m]

        return [grab(m) for m in new_matches]
    return extract_fiducials

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