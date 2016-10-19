'''
SIMRecon Utility Functions
'''
# import some os functionality so that we can be platform independent
import os
import glob
import re
import tqdm
import numpy as np
import matplotlib.pyplot as plt
# need subprocess to run commands
import subprocess
import hashlib
import shutil
# import our ability to read and write MRC files
import Mrc

from collections import OrderedDict, Sequence
# import skimage components
from peaks.peakfinder import PeakFinder

from dphutils import (slice_maker, scale_uint16, fft_pad,
                      nextpow2, radial_profile)
from dphplotting import display_grid
from pyOTF.phaseretrieval import *
from pyOTF.utils import *
try:
    from pyfftw.interfaces.numpy_fft import (fftshift, ifftshift, fftn, ifftn,
                                             rfftn, fftfreq, rfftfreq)
    import pyfftw
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import fftshift, ifftshift, fftn, ifftn, rfftn, fftfreq, rfftfreq
from skimage.external import tifffile as tif


class PSFFinder(object):
    """Object to find and analyze subdiffractive emmitters"""

    def __init__(self, stack, psfwidth=1.3, window_width=20, **kwargs):
        """Analyze a z-stack of subdiffractive emmitters

        Parameters
        ----------
        stack : ndarray


        Kwargs
        ------
        psfwidth : float
        window_width : int"""
        self.stack = stack
        self.peakfinder = PeakFinder(stack.max(0), psfwidth, **kwargs)
        self.peakfinder.find_blobs()
        self.all_blobs = self.peakfinder.blobs
        self.window_width = window_width
        self.find_psfs(2 * psfwidth)

    def find_psfs(self, max_s=2.1, num_peaks=20):
        """Function to find and fit blobs in the max intensity image

        Blobs with the appropriate parameters are saved for further fitting.

        Parameters
        ----------
        max_s: float
            Reject all peaks with a fit width greater than this
        num_peaks: int
            The number of peaks to analyze further"""
        window_width = self.window_width
        # pull the PeakFinder object
        my_PF = self.peakfinder
        # find blobs
        my_PF.find_blobs()
        # prune blobs
        my_PF.remove_edge_blobs(window_width)
        my_PF.prune_blobs(window_width)
        # fit blobs in max intensity
        blobs_df = my_PF.fit_blobs(window_width)
        # round to make sorting a little more meaningfull
        blobs_df.SNR = blobs_df.dropna().SNR.round().astype(int)
        # sort by SNR then sigma_x after filtering for unreasonably
        # large blobs and reindex data frame here
        new_blobs_df = blobs_df[
            blobs_df.sigma_x < max_s
        ].sort_values(
            ['SNR', 'sigma_x'], ascending=[False, True]
        ).reset_index(drop=True)
        # set the internal state to the selected blobs
        my_PF.blobs = new_blobs_df[
            ['y0', 'x0', 'sigma_x', 'amp']
        ].values.astype(int)
        self.fits = new_blobs_df

    def find_window(self, blob_num=0):
        """Finds the biggest window distance."""
        # pull all blobs
        blobs = self.all_blobs
        # three different cases
        if not len(blobs):
            # no blobs in window, raise hell
            raise RuntimeError("No blobs found, can't find window")
        else:
            # TODO: this should be refactored to use KDTrees
            # more than one blob find
            best = np.round(
                self.fits.iloc[blob_num][['y0', 'x0', 'sigma_x', 'amp']].values
            ).astype(int)

            def calc_r(blob1, blob2):
                """Calc euclidean distance between blob1 and blob2"""
                y1, x1, s1, a1 = blob1
                y2, x2, s2, a2 = blob2
                return np.sqrt((y1 - y2)**2 + (x1 - x2)**2)
            # calc distances
            r = np.array([calc_r(best, blob) for blob in blobs])
            # find min distances
            # remember that best is in blobs so 0 will be in the list
            # find the next value
            r.sort()
            try:
                r_min = r[1]
            except IndexError:
                # make r_min the size of the image
                r_min = min(
                    np.concatenate((np.array(self.stack.shape[1:3]) - best[:2],
                                    best[:2]))
                )
            # now window size equals sqrt or this
            win_size = int(round(2 * (r_min / np.sqrt(2) - best[2] * 3)))

        window = slice_maker(best[0], best[1], win_size)
        self.window = window

        return window

    def plot_all_windows(self):
        """Plot all the windows so that user can choose favorite"""
        windows = [self.find_window(i) for i in range(len(self.fits))]
        fig, axs = display_grid({i: self.peakfinder.data[win]
                                 for i, win in enumerate(windows)})
        return fig, axs


class PhaseRetriever(PSFFinder):
    """Utility to phase retrieve mutltiple emitters in a single dataset"""

    def __init__(self, stack, wl, na, ni, res, zres, **kwargs):
        """"""
        psfwidth = wl / 4 / na / res
        super().__init__(stack, psfwidth, **kwargs)
        # initialize model params
        self.params = dict(
            na=na,
            ni=ni,
            wl=wl,
            res=res,
            zres=zres,
            zsize=stack.shape[0]
        )

    def retrieve_phase(self, blob_num=0, xysize=None, **kwargs):
        """"""
        window = self.find_window(blob_num)
        data = self.stack[[Ellipsis] + window]
        data_prepped = prep_data_for_PR(data, xysize)
        self.params.update(dict(size=data_prepped.shape[-1]))
        self.pr_result = retrieve_phase(data_prepped, self.params, **kwargs)
        return self.pr_result


class SIMOTFMaker(PSFFinder):

    def __init__(self, stack, na=0.85, pixsize=0.13,
                 det_wl=0.585, **kwargs):
        """Find PSFs and turn them into OTFs

        Parameters
        ----------
        stack : ndarray
        na : float
        pixsize : float
        det_wl : float
        """
        psfwidth = det_wl / 4 / na / pixsize
        super().__init__(stack, psfwidth, **kwargs)
        self.na = na
        self.pixsize = pixsize
        self.det_wl = det_wl

    def plot_window(self, blob_num, **kwargs):
        """Plot all the things for this window"""
        self.find_window(blob_num)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.matshow(self.peakfinder.data[self.window])
        self.gen_radialOTF(show_OTF=True, **kwargs)
        ax2.semilogy(abs(self.radprof))
        fig.tight_layout()

    def calc_infocus_psf(self, filter_kspace=True, filter_xspace=True):
        '''
        Calculate the infocus psf
        '''
        img_raw = self.stack[[Ellipsis] + self.window]
        img_raw = remove_bg(img_raw, 1.0)
        # pad image
        psf = fft_pad(img_raw)
        # recenter
        # TODO: add this part
        # fft
        otf = fftshift(fftn(ifftshift(psf))).mean(0)
        # filter in k-space, if requested
        if filter_kspace or filter_xspace:
            yy, xx = np.indices(otf.shape[-2:]) - np.array(otf.shape[-2:])[:, np.newaxis, np.newaxis] / 2
            r = np.hypot(yy, xx)
        if filter_kspace:
            # need to multiply be 1000 for the wavelength conversion (nm -> um)
            # and need to convert pixel size to k-space, equivalent
            mask = r > (2 * self.na / self.det_wl * self.pixsize) * otf.shape[-1]
            otf[mask] = 0
        # ifft
        infocus_psf = abs(fftshift(ifftn(ifftshift(otf))))
        # filter in x-space if requested
        if filter_xspace:
            mask = r > 4 * (self.det_wl / 2 * self.na / self.pixsize)
            infocus_psf[mask] = 0
        self.psf = infocus_psf

    def gen_radialOTF(self, lf_cutoff=0.1, width=3, **kwargs):
        """Generate the Radially averaged OTF from the sample data."""
        img_raw = self.stack[[Ellipsis] + self.window]
        img_raw = remove_bg(img_raw, 1.0)
        if img_raw.shape[-1] < 512 or img_raw.shape[-2] < 512:
            img = fft_pad(img_raw, (nextpow2(img_raw.shape[0]), 512, 512))
        else:
            img = fft_pad(img_raw)
        # pull the max x, y size
        nx = max(img.shape[-1:-3:-1])
        # calculate the um^-1/pix
        dkr = 1 / (nx * self.pixsize)
        # save dkr for later
        self.dkr = dkr
        # calculate the kspace cutoff, round up (that's what the 0.5 is for)
        krcutoff = int(2 * self.na / (self.det_wl) / dkr + .5)
        radprof = calc_radial_OTF(img, krcutoff, **kwargs)
        if lf_cutoff is not None:
            # if given a cutoff linearly fit the points around it to a line
            # then interpolate the line back to the origin starting at the low
            # frequency
            # find the cutoff in terms of pixels
            mid_num = int(lf_cutoff / dkr + .5)
            # choose the low frequency
            lf = mid_num - width
            if lf > 0:
                # if the low frequency is higher than the DC component then
                # proceed with the fit, we definitely don't want to include
                # the DC
                hf = mid_num + width
                m, b = np.polyfit(np.arange(lf, hf), radprof[lf:hf], 1)

                radprof[:mid_num] = np.arange(0, mid_num) * m + b

            else:
                # set DC to mid_num to mid_num
                radprof[:mid_num] = radprof[mid_num]
            # TODO: add bit to remove average phase angle from profile.
            radprof /= radprof.max()

        self.radprof = radprof

        print('Better cutoff is {:.3f}'.format(
            (abs(self.radprof[:krcutoff]).argmin() -
             1) / (2 / (self.det_wl) / self.dkr)))

    def save_radOTF_mrc(self, output_filename, **kwargs):
        # make empty header
        header = Mrc.makeHdrArray()
        # initialize it
        # set type and shape
        Mrc.init_simple(header, 4, self.radprof.shape)
        # set wavelength
        header.wave = self.det_wl * 1000
        # set number of wavelengths
        header.NumWaves = 1
        # set dimensions
        header.d = (self.dkr,) * 3
        tosave = self.radprof.astype(np.complex64)
        # save it
        tosave = tosave.reshape(1, 1, (len(tosave)))

        Mrc.save(tosave, output_filename, hdr=header, **kwargs)

    def save_PSF_mrc(self, output_filename):
        '''
        Object specific wrapper for general save_PSF_mrc
        '''

        # take the best blob and pad to at least 512
        img_raw = self.best_blob_data

        if img_raw.shape[0] < 512:
            img = fft_pad(img_raw, 512)
        else:
            img = fft_pad(img_raw)

        save_PSF_mrc(img, output_filename, self.pixsize, self.det_wl)


def save_PSF_mrc(img, output_filename, pixsize=0.0975, det_wl=520):
    '''
    A small utility function to save an image of a bead as an MRC

    Parameters
    ----------
    img: ndarray, rank 2
        The image to save
    output_filename: path
        The filename to output to
    pixsize: float
        the the pixel size in microns (size of the sensor pixel at the sample)
    det_wl: float
        the detection wavelength
    '''

    # TODO: make sure '.mrc' is appended to files that don't have it.
    from pysegtools.mrc import MRC

    ny, nx = img.shape
    PSFmrc = MRC(output_filename, nx=nx, ny=ny, dtype=img.dtype)
    PSFmrc.header['nz'] = 1
    PSFmrc[0] = img
    PSFmrc.header['nwave'] = 1       # detection wavelength
    PSFmrc.header['wave1'] = det_wl  # detection wavelength
    # need the rest of these fields filled out otherwise header won't write.
    PSFmrc.header['wave2'] = 0
    PSFmrc.header['wave3'] = 0
    PSFmrc.header['wave4'] = 0
    PSFmrc.header['wave5'] = 0
    # fill in the pixel size
    PSFmrc.header['xlen'] = pixsize
    PSFmrc.header['ylen'] = pixsize

    # need to delete this field to let MRC know that this is an oldstyle
    # header to write
    del PSFmrc.header['cmap']

    # write the header and close the file.
    PSFmrc.write_header()
    PSFmrc.close()

    return output_filename


def calc_radial_mrc(infile, outfile=None, na=0.85, L=8, H=22):
    '''
    A simple wrapper around the radial OTF calc
    '''

    # TODO: Error checking
    # make sure we have the absolute path
    infile = os.path.abspath(infile)
    if outfile is None:
        outfile = infile.replace('.mrc', '_otf2d.mrc')
    else:
        outfile = os.path.abspath(outfile)

    # write our string to send to the shell
    # 8 is the lower pixel and 22 is the higher pixel
    # 0.8 is the detection na
    # otfcalc = r'C:\newradialft\otf2d -N {na} -L {L} -H {H} {infile} {outfile}'

    # format the string
    # excstr = otfcalc.format(infile=infile, outfile=outfile, na=na, L=L, H=H)
    # send to shell
    # os.system(excstr)

    return_code = subprocess.call([
                                  r'C:\newradialft\otf2d',
                                  '-N', str(na),
                                  '-L', str(L),
                                  '-H', str(H),
                                  infile,
                                  outfile
                                  ])

    return return_code

# Here lie all the bits for processing a 3D otf, this code I'm not proud of at
# all but it seems to work


def center_data(data, max_loc=None):
    """Utility to center the data

    Parameters
    ----------
    data : ndarray
        Array of data points

    Returns
    -------
    centered_data : ndarray same shape as data
        data with max value at the central location of the array
    """
    if max_loc is None:
        max_loc = np.unravel_index(data.argmax(), data.shape)
    # copy data
    centered_data = data.copy()
    # extract shape and max location
    data_shape = data.shape
    # iterate through dimensions and roll data to the right place
    for i, (x0, nx) in enumerate(zip(max_loc, data_shape)):
        if x0 is not None:
            centered_data = np.roll(centered_data, nx // 2 - x0, i)
    return centered_data


def makematrix(nphases, norders):
    """Make a separation matrix

    See Gustafsson 2008 Bio Phys J"""
    # phi is the phase step
    phi = 2 * np.pi / nphases
    # number of columns is determined by the number of orders
    cols = (norders * 2 + 1)
    # prepare an empty array to fill
    # this is very poor numpy coding but this will never be the bottleneck
    sep_mat = np.zeros(nphases * cols)
    # fill the matrix
    for j in range(nphases):
        sep_mat[0 * nphases + j] = 1.0
        for order in range(1, norders + 1):
            sep_mat[(2 * order - 1) * nphases + j] = np.cos(j * order * phi)
            sep_mat[2 * order * nphases + j] = np.sin(j * order * phi)
    return sep_mat.reshape(nphases, cols)


def rescale(otf):
    """Takes a radially averaged otf and scales by max of 0th order"""
    # assume OTF has arrangement: order, z, r
    scale_factor = abs(otf[0]).max()
    return otf / scale_factor


def rfft_trunk(nx):
    """just to test function below"""
    return nx // 2 + 1


def irfft_trunk(nx):
    """undo the trunkation along the real axis when performing an rfft"""
    return (nx - nx % 2) * 2


def test_rfft_trunc():
    """Test forward and backward truncs"""
    assert 0 == irfft_trunk(0)
    kx = 512
    assert kx == irfft_trunk(rfft_trunk(kx))


def _kspace_coords(dz, dr, shape):
    """Make the kspace coordinates for a radially averaged OTF

    Right now the function assumes that the extent of dr is the extent of dx"""
    # split open shape
    nz, nr = shape
    # calculate kz, kr freqs, assume that all data has been shifted back
    kz = fftshift(fftfreq(nz, dz))
    kr = rfftfreq(irfft_trunk(nr), dr)
    # determine delta kz
    dkz = kz[1] - kz[0]
    dkr = kr[1] - kr[0]
    # make the grids
    krr, kzz = np.meshgrid(kr, kz)
    return kzz, krr, dkz, dkr


def makemask(wl, na, ni, dz, dr, shape, offset=(0, 0), expansion=0):
    """Make a mask for the 3D OTF radially averaged

    Parameters
    ----------
    wl : float
        Wavelength of emission of data in nm
    na : float
        Numerical aperature of objective
    ni : float
        Index of refraction of media
    dz : float
        z resolution in nm
    dr : float
        x/y resolution, in nm
    shape : tuple
        shape of data
    offset : tuple
        The offset of the mask, in units of pixels
    expansion : float
        The expansion (or contraction) of the mask, in pixels

    Returns
    -------
    mask : ndarray
        A boolean array of shape `shape` that masks the otf outside its
        theoretical support
    """
    # unpack offsets
    z_offset, r_offset = offset
    # make k-space coordinates
    kz, kr, dkz, dkr = _kspace_coords(dz, dr, shape)
    kr_max = ni / wl  # the radius of the spherical shell
    kr_0 = na / wl  # the offset of the circle from the origin
    # z displacement of circle's center
    z0 = np.sqrt(kr_max ** 2 - kr_0 ** 2)
    # expand mask if requested
    kr_max += dkr * expansion
    # calculate centered kr
    cent_kr = kr - kr_0 - r_offset * dkr
    # calculate top half
    onehalf = np.hypot(cent_kr, kz - z0 - z_offset * dkz) <= kr_max
    # calculate bottom half
    otherhalf = np.hypot(cent_kr, kz + z0 - z_offset * dkz) <= kr_max
    mask = np.logical_and(otherhalf, onehalf)
    return mask


def find_origin(rad_prof):
    """Find the origin in a radially averaged profile"""
    # copy data, otherwise it will be changed
    data = rad_prof.copy()
    # grab shape
    nz, nr = data.shape
    # suppress pure DC
    data[(nz + 1) // 2, 0] = 0
    # find new max
    max_loc = np.unravel_index(abs(data).argmax(), data.shape)
    # only return kz component
    return max_loc[0]


def find_offsets(rad_prof):
    """find the offsets of the OTF bands"""
    # find potential origin
    dz = find_origin(rad_prof)
    nz, nr = rad_prof.shape
    center = nz // 2
    if dz == center:
        # if potential origin is same as center return zero
        return (0,)
    else:
        # else assume we have a band with two fold symmetry and we
        # should return the above and below parts
        return dz - center, center - dz


def mask_rad_prof(rad_prof, exp_args):
    """Mask off the radial profiles according to the experimental arguments

    Returns both the mask and the masked radial profile"""
    offsets = find_offsets(rad_prof)
    masks = np.array([makemask(*exp_args, rad_prof.shape, offset=(o, 0))
                      for o in offsets])
    mask = np.logical_or.reduce(masks, 0)
    return rad_prof * mask, mask


def correct_phase_angle(band, mask):
    """Remove the average phase angle"""
    # norm all values so they're on the unit circle
    # only use valid values within otf theoretical support
    valid_values = band[mask]
    assert valid_values.size, "There are not valid values!"
    normed = valid_values / abs(valid_values)
    # calculate angle of average
    phi = np.angle(normed.mean())
    # shift band to remove average phase angle
    a = band * np.exp(-1j * phi)
    # set everything outside suppor to zero
    a[np.logical_not(mask)] = 0
    return a


def average_pm_kz(data):
    """Average the top and bottom of the otf which should be conjugate
    symmetric"""
    # assume data has kz axis first
    nz = data.shape[0]
    # TODO: this part needs work!
    if nz % 2:
        # nz is odd
        data_top = data[:nz // 2]
        # + 1 here in case nz odd
        data_bottom = data[nz // 2 + 1:]
        data_avg = (data_top + np.conj(data_bottom[::-1])) / 2
        return np.concatenate((data_avg, data[np.newaxis, nz // 2],
                               np.conj(data_avg[::-1])))
    else:
        # nz is even
        data_top = data[1:nz // 2]
        # + 1 here in case nz odd
        data_bottom = data[nz // 2 + 1:]
        data_avg = (data_top + np.conj(data_bottom[::-1])) / 2
        return np.concatenate((data[np.newaxis, 0], data_avg,
                               data[np.newaxis, nz // 2],
                               np.conj(data_avg[::-1])))


class PSF3DProcessor(object):
    """An object designed to turn a 3D SIM PSF into a 3D SIM radially averaged
    OTF"""

    def __init__(self, data, exp_args):
        """Initialize the object, assumes data is already organized as:
        directions, phases, z, y, x

        exp_args holds all the experimental parameters (should be dict):
        wl, na, ni, zres, rres"""
        # set up internal data
        self.data = data
        # extract experimental args
        self.exp_args = self.wl, na, ni, dz, dr = exp_args
        # get ndirs etc
        self.ndirs, self.nphases, self.nz, self.ny, self.nx = data.shape
        # remove background
        self.data_nobg = data_nobg = remove_bg(self.data, 1.0)
        # average along directions and phases to make widefield psf
        self.conv_psf = conv_psf = data_nobg.mean((0, 1))
        # separate data
        sep_data = self.separate_data()
        # center the data using the conventional psf center
        psf_max_loc = np.unravel_index(conv_psf.argmax(), conv_psf.shape)
        cent_data = center_data(sep_data, (None, ) + psf_max_loc)
        # take rfft along spatial dimensions (get seperated OTFs)
        # last fftshift isn't performed along las axis, because it's the real
        # axis
        self.cent_data_fft_sep = fftshift(rfftn(ifftshift(
            cent_data, axes=(1, 2, 3)), axes=(1, 2, 3)), axes=(1, 2)
        )
        self.avg_and_mask()
        # get spacings and save for later
        kzz, krr, self.dkz, self.dkr = _kspace_coords(dz, dr, self.masks[0].shape)
        # average bands (hard coded for convenience)
        corrected_profs = np.array([
            correct_phase_angle(b, m)
            for b, m in zip(self.masked_rad_profs, self.masks)
        ])
        band0 = corrected_profs[0]
        band1 = (corrected_profs[1] + corrected_profs[2]) / 2
        band2 = (corrected_profs[3] + corrected_profs[4]) / 2
        self.bands = np.array((band0, band1, band2))
        self.bands = np.array([average_pm_kz(band) for band in self.bands])

    def separate_data(self):
        """Separate the different bands"""
        # make the separation matrix and apply it
        sep_mat = makematrix(self.nphases, self.nphases // 2)
        # add extra axis to "store" the linear combinations of the "vectors"
        # (we really need to sum along this axis to get the linear combos)
        # sum the linear combinations and take mean along directions
        # to make the broadcasting work we need to expand out the separation
        # matrix now too. The data ordering is now:
        # dirs, combos, phases, z, y, x
        sep_mat = sep_mat[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]
        # this multiplication recombines each phase image
        temp_data = (self.data_nobg[:, np.newaxis] * sep_mat)
        # sum the weighted images (linear combination) and take mean along
        # directions
        self.sep_data = temp_data.sum(2).mean(0)
        return self.sep_data

    def avg_and_mask(self):
        # radially average the OTFs
        # for each otf in the seperated data and for each kz plane calculate
        # the radial average center the radial average at 0 for last axis
        # because of real fft
        center = ((self.ny + 1) // 2, 0)
        extent = self.nx // 2 + 1
        self.r_3D = r_3D = np.array([
            [radial_profile(o, center)[0][:extent]
             for o in z] for z in self.cent_data_fft_sep
        ])
        # mask OTFs and retrieve masks
        self.masked_rad_profs, masks = np.swapaxes(
            np.array([mask_rad_prof(r, self.exp_args) for r in r_3D]), 0, 1
        )
        # convert masks to bool (they've been cast to complex in the above)
        self.masks = masks.astype(bool)

    def save_radOTF_mrc(self, output_filename, **kwargs):
        # make empty header
        header = Mrc.makeHdrArray()
        # initialize it
        # set type and shape
        Mrc.init_simple(header, 4, self.bands.shape)
        # set wavelength
        header.wave = self.wl
        # set number of wavelengths
        header.NumWaves = 1
        # set dimensions
        header.d = (self.dkz * 1000, self.dkr * 1000, 0.0)
        bands = swapaxes(self.bands, 1, 2)
        bands = rescale(ifftshift(bands, axes=2))
        tosave = bands.astype(np.complex64)

        Mrc.save(tosave, output_filename, hdr=header, **kwargs)


def simrecon(*, input_file, output_file, otf_file, **kwargs):
    '''
    A simple wrapper to Lin's sirecon.exe

    Parameters
    ----------
    input_file: path
        Path to file holding raw SIM data
    output_file: path
        Path to location to write reconstruction
    OTF_file: path
        Path to OTF file to use in reconstruction

    Options
    -------
    ndirs: int (default is 3)
        number of directions in data
    nphases: int (default is 5)
        number of phases in data
    2lenses: bool
        data acquired with 2 opposing objectives
    bessel: bool
        data acquired with Bessel beam SIM
    fastSIM: bool
        data acquired with fast live SIM, i.e. data organized into (nz, ndirs, nphases)
    recalcarray: int (default is 1)
        how many times do you want to re-calculuate overlapping arrays
    inputapo: int
        number of pixels to apodize the input data (-1 to cosine apodize)
    forcemodamp: sequence of floats (f1 f2... f_norders)
        force the modulation amplitude to be f1 and f2
                If other than 3 phases are used, the -nphases flag must be used
                 BEFORE the -forcemodamp flag
    nok0search: bool
        do not want to search for the best k0
    nokz0: bool
        do not use kz0 plane in makeoverlaps() or assembly (for bad
    k0searchAll: bool
        search for k0 for every time point in a time series
    k0angles: sequence of floats (f0 f1... f_(ndirs-1))
        user supplied pattern angle list, the -ndirs flag must be used BEFORE
        the -k0angles flag
    fitonephase: bool
        in 2D NLSIM for orders > 1, modamp's phase will be order 1 phase
        multiplied by order; default is using fitted phases for all orders
    noapodizeout: bool
        do not want to apodize the output data
    gammaApo: float
        apodize the output data with a power function
    preciseapo: bool
        Apply precise apo or not
    zoomfact: float
        factor by which to subdivide pixels laterally
    zzoom: float
        factor by which to subdivide pixels axially
    zpadto: int
        how many total sections after zero padding
    explodefact: float
        factor by which to exaggerate the order shifts for display
    nofilteroverlaps: bool (default True)
        (Used with explodefact) leave orders round
        (no filtering the overlapped regions)
    nosuppress: bool
        do not want to suppress singularity at OTF origins
    suppressR: float
        the radius of range
    dampenOrder0: bool
        dampen order 0 in filterbands
    noOrder0: bool
        do not use order 0 in assembly
    noequalize: bool
        no equalization of input data
    equalizez: bool
        to equalize images of all z sections and directions
    equalizet: bool
        to equalize all time points based on the first one
    wiener: float (default 0.01)
        set wiener constant
    wienerInr: float (default is 0.00)
        wiener constant of the final time point will be wiener + this number
    background: float (default is 515)
        set the constant background intensity
    bgInExtHdr: bool
        the background of each section is recorded in the extended header's 3rd
        float number (in Lin's scope)
    otfRA: bool
        to use radially averaged OTFs
    driftfix: bool
        to estimate and then fix drift in 3D
    driftHPcutoff: float
        the cutoff frequency (fraction of lateral resolution limit) used in
        high-pass Gaussian filter in determining drift (default 0.0)
    fix2Ddrift: bool
        to correct 2D drifts for each exposure within a pattern direction
    fixphasestep: bool
        to correct phase used in separation matrix based on within-direction
        drift correction
    noff: int
        number of switch-off images in nonlinear SIM
    usecorr: path
        use correction file to do flatfielding
    nordersout: int
        the number of orders to use in reconstruction. Use if different from
        (n_phases+1)/2
    angle0: float
        the starting angle (in radians) of the patterns
    negDangle: bool
        use negative angle step
    ls: float
        the illumination pattern's line spacing (in microns)
    na: float
        the (effective) na of the objective
    nimm: float
        the index of refraction of the immersion liquid
    saveprefiltered: path
        save separated bands into file
    savealignedraw: path
        save drift-corrected raw images into file
    saveoverlaps: path
        save overlaps by makeoverlaps() into file
    '''

    # make sure the paths are absolute paths
    input_file = os.path.abspath(input_file)
    output_file = os.path.abspath(output_file)
    otf_file = os.path.abspath(otf_file)
    for file in (input_file, otf_file):
        assert os.path.exists(file), "{} doesn't exist!".format(file)
    # the list to pass to subprocess.call, this is just the beginning
    exc_list = [r'C:\SIMrecon_svn\sirecon', input_file, output_file, otf_file]
    # insert default values into **kwargs here
    valid_kwargs = OrderedDict.fromkeys((
        'ndirs',
        'nphases',
        '2lenses',
        'bessel',
        'fastSIM',
        'recalcarray',
        'inputapo',
        'nordersout',
        'forcemodamp',
        'nok0search',
        'nokz0',
        'k0searchAll',
        'k0angles',
        'fitonephase',
        'noapodizeout',
        'gammaApo',
        'preciseapo',
        'zoomfact',
        'zzoom',
        'zpadto',
        'explodefact',
        'nofilteroverlaps',
        'nosuppress',
        'suppressR',
        'dampenOrder0',
        'noOrder0',
        'noequalize',
        'equalizez',
        'equalizet',
        'wiener',
        'wienerInr',
        'background',
        'bgInExtHdr',
        'otfRA',
        'driftfix',
        'driftHPcutoff',
        'fix2Ddrift',
        'fixphasestep',
        'noff',
        'usecorr',
        'angle0',
        'negDangle',
        'ls',
        'na',
        'nimm',
        'saveprefiltered',
        'savealignedraw',
        'saveoverlaps'
    ))
    numeric = (int, float)
    valid_kwargs.update({
        'ndirs': int,
        'nphases': int,
        'phaselist': Sequence,
        '2lenses': bool,
        'bessel': bool,
        'fastSIM': bool,
        'recalcarray': int,
        'inputapo': int,
        'forcemodamp': Sequence,
        'nok0search': bool,
        'nokz0': bool,
        'k0searchAll': bool,
        'k0angles': Sequence,
        'fitonephase': bool,
        'noapodizeout': bool,
        'gammaApo': float,
        'preciseapo': bool,
        'zoomfact': numeric,
        'zzoom': float,
        'zpadto': int,
        'explodefact': float,
        'nofilteroverlaps': bool,
        'nosuppress': bool,
        'suppressR': numeric,
        'dampenOrder0': bool,
        'noOrder0': bool,
        'noequalize': bool,
        'equalizez': bool,
        'equalizet': bool,
        'wiener': float,
        'wienerInr': float,
        'background': numeric,
        'bgInExtHdr': bool,
        'otfRA': bool,
        'driftfix': bool,
        'driftHPcutoff': float,
        'fix2Ddrift': bool,
        'fixphasestep': bool,
        'noff': int,
        'usecorr': 'path',
        'nordersout': int,
        'angle0': float,
        'negDangle': bool,
        'ls': float,
        'na': float,
        'nimm': float,
        'saveprefiltered': 'path',
        'savealignedraw': 'path',
        'saveoverlaps': 'path'
    })

    # update kwargs with those passed by user and generate the list.
    for k, kw_type in valid_kwargs.items():
        try:
            kw_value = kwargs[k]
        except KeyError:
            # user didn't pass this one, so skip
            pass
        else:
            # test validity
            if kw_type == 'path':
                # assert os.path.exists(kw_value), '{} is an invalid path'.format(kw_value)
                pass
            else:
                assert isinstance(kw_value, kw_type), '{} is type {} and should have been type {}'.format(k, type(kw_value), repr(kw_type))
            if kw_type is not bool:
                exc_list.append('-' + k)
                # exc_list.append('-' + k)
                if isinstance(kw_value, Sequence) and not isinstance(kw_value, str):
                    for v in kw_value:
                        exc_list.append(formatter(v))
                else:
                    exc_list.append(formatter(kw_value))
            else:
                # in this case the key word is a bool
                # test if bool is true
                if kw_value:
                    exc_list.append('-' + k)

    # save the output
    return_code = subprocess.check_output(exc_list)

    return return_code.decode('utf-8').split('\n')

def formatter(value):
    if isinstance(value, float):
        return "{:.16f}".format(value)
    else:
        return str(value)


def write_mrc(input_file):
    raise NotImplementedError


def calc_radial_OTF(psf, krcutoff=None, show_OTF=False):
    '''
    Calculate radially averaged OTF given a PSF and a cutoff value.

    This is designed to work well with Lin's SIMRecon software

    Parameters
    ----------
    psf: ndarray, 2-dim, real
        The psf from which to calculate the OTF
    krcutoff: int
        The diffraction limit in pixels.

    Returns
    -------
    radprof: ndarray, 1-dim, complex
        Radially averaged OTF
    '''
    # assumes background has already been removed from PSF
    # recenter
    # TODO: add this part

    # fft, switch to rfft
    otf = fftshift(fftn(ifftshift(newpsf)))

    if show_OTF:
        from dphplotting import mip
        from matplotlib.colors import LogNorm
        # this is still wrong, need to do the mean before summing
        # really we need a slice function.
        mip(abs(otf), func=np.mean, norm=LogNorm())

    if otf.ndim > 2:
        # if we have a 3D OTF collapse it by summation along kz into a 2D OTF.
        otf = otf.mean(0)

    center = np.array(otf.shape) / 2

    radprof, _ = radial_profile(otf)[:int(center[0] + 1)]

    radprof /= radprof.max()

    if krcutoff is not None:
        # calculate mean phase angle of data within diffraction limit
        temp = radprof[:krcutoff]
        temp /= abs(temp)
        phi = np.angle(mean(temp))
        # remove mean phase angle
        radprof *= np.exp(-1j * phi)
        # set everything beyond the diffraction limit to 0
        radprof[krcutoff:] = 0
    # return
    return radprof


def crop_mrc(fullpath, window=None, extension='_cropped'):
    '''
    Small utility to crop MRC files

    Parameters
    ----------
    fullpath : path
        path to file
    window : slice (optional)
        crop window

    Returns
    -------
    croppath : path
        path to cropped file
    '''
    # open normal MRC file
    oldmrc = Mrc.Mrc(fullpath)
    old_data = oldmrc.data
    # make the crop path
    croppath = fullpath.replace('.mrc', extension + '.mrc')
    # crop window
    if window is None:
        nz, ny, nx = old_data.shape
        window = [slice(None, None, None)] + slice_maker(ny // 2, nx // 2,
                                                         max(ny, nx) // 2)
    # prepare a new file to write to
    Mrc.save(old_data[window], croppath, ifExists='overwrite', hdr=oldmrc.hdr)
    # close the old MRC file.
    oldmrc.close()
    del oldmrc
    return croppath

# split, process, recombine functions


def split_img(img, side):
    '''
    A utility to split a SIM stack into substacks
    '''

    # Testing input
    divisor = img.shape[-1] // side
    # Error checking
    assert side == img.shape[-1] / divisor, 'Side {}, not equal to {}/{}'.format(
        side, img.shape[-1], divisor)
    assert img.shape[-2] == img.shape[-1]
    assert img.shape[-1] % divisor == 0

    # reshape array so that it's a tiled image
    img_s0 = img.reshape(-1, divisor, side, divisor, side)
    # roll one axis so that the tile's y, x coordinates are next to each other
    img_s1 = np.rollaxis(img_s0, 3, 1)
    # combine the tile's y, x coordinates into one axis.
    img_s2 = img_s1.reshape(-1, divisor**2, side, side)
    # roll axis so that we can easily iterate through tiles
    return np.rollaxis(img_s2, 1, 0)


def combine_img(img_stack):
    '''
    A utility to combine a processed stack.
    '''

    num_sub_imgs, ylen, xlen = img_stack.shape
    divisor = int(np.sqrt(num_sub_imgs))
    assert xlen == ylen, '{} != {}'.format(xlen, ylen)
    assert np.sqrt(num_sub_imgs) == divisor

    return np.rollaxis(
        img_stack.reshape(divisor, divisor, ylen, xlen), 0, 3
    ).reshape(ylen * divisor, xlen * divisor)


def split_img_with_padding(img, side, pad_width, mode='reflect'):
    '''
    Split SIM stack into sub-stacks with padding of pad_width
    '''
    # if no padding revert to simpler function.
    if pad_width == 0:
        return split_img(img, side)
    # pull the shape of the image
    # NOTE: need to refactor this so that it works well with
    # nt, np, ny, nx waves
    # nall = img.shape[:-2]
    # ny, nx = img.shape[-2], img.shape[-1]
    nz, ny, nx = img.shape
    # make sure the sides are equal
    assert nx == ny, "Sides are not equal, {} != {}".format(nx, ny)
    # make sure that side cleanly divides img dimensions
    assert nx % side == 0, "Sides are not mutltiples of tile size, {} % {} != 0".format(nx, side)
    # pad the whole image
    # pad_img = fft_pad(img, nall + (pad_width + ny, pad_width + nx), mode)
    pad_img = fft_pad(img, (nz, pad_width + ny, pad_width + nx), mode)
    # split the image into padded sub-images
    split_pad_img = np.array([pad_img[...,
                                      j * side:pad_width + (j + 1) * side,
                                      i * side:pad_width + (i + 1) * side]
                              for i in range(nx // side)
                              for j in range(nx // side)])
    # return this
    return split_pad_img


def cosine_edge(pad_size):
    '''
    Generates a cosine squared edge

    When added to its reverse it equals 1

    Parameters
    ----------
    pad_size : int
        The size of the edge (i.e. the amount of image padding)

    Returns
    -------
    edge : ndarray (1D)
        The array representing the edge

    Example
    -------
    >>> edge = cosine_edge(10)
    >>> rev_edge = edge[::-1]
    >>> np.allclose(edge + rev_edge, np.ones_like(edge))
    True
    '''
    x = np.arange(pad_size)
    return np.sin(np.pi * x / (pad_size - 1) / 2)**2


def linear_edge(pad):
    '''
    Generates a linear edge

    When added to its reverse it equals 1

    Parameters
    ----------
    pad_size : int
        The size of the edge (i.e. the amount of image padding)

    Returns
    -------
    edge : ndarray (1D)
        The array representing the edge

    Example
    -------
    >>> edge = linear_edge(10)
    >>> rev_edge = edge[::-1]
    >>> np.allclose(edge + rev_edge, np.ones_like(edge))
    True
    '''
    return np.arange(pad)/(pad-1)


def edge_window(center_size, edge_size, window_func=cosine_edge):
    '''
    Generate a 1D window that ramps up through the padded region and is flat in
    the middle

    Parameters
    ----------
    center_size : int
        The size of the center part
    edge_size : int
        The size of the edge parts

    Returns
    -------
    edge_window : ndarray (1D)
        a window with a rising and falling edge
    '''
    center_part = np.ones(center_size)
    left_part = window_func(edge_size)
    right_part = left_part[::-1]
    return np.concatenate((left_part, center_part, right_part))


def extend_and_window_tile(tile, pad_size, tile_num, num_tiles,
                           window_func=cosine_edge):
    '''
    Function that takes a tile that has been padded and its tile number and
    places it in the correct space of the overall image and windows the
    function before padding with zeros

    Parameters
    ----------
    tile : ndarray (img)
        the tile to pad and window
    pad_size : int
        the amount of padding in the tile
    tile_num : int
        Assumes data is from a `split_img` operation, this is the index
    num_tiles : int
        total number of tiles
    window_func : cosine_edge (default, callable)
        the window function to use

    Returns
    -------
    tile : ndarray
        A tile that has been windowed and then extended to the appropriate size
    '''
    # calculate the total number of tiles in the x and y directions
    # (assumes square image)
    ytot = xtot = int(np.sqrt(num_tiles))
    assert ytot * xtot == num_tiles, "Image is not square!"
    # calculate the position of this tile
    yn, xn = tile_num % ytot, tile_num // ytot
    # calculate the unpadded size of the tile (original tile size)
    to_pad = (tile.shape[-1] - pad_size)
    # calculate the before and after padding for each direction
    ybefore = yn * to_pad
    yafter = (ytot - yn - 1) * to_pad
    xbefore = xn * to_pad
    xafter = (xtot - xn - 1) * to_pad
    # make y window and x window
    ywin = edge_window(to_pad - pad_size, pad_size, window_func=window_func)
    xwin = edge_window(to_pad - pad_size, pad_size, window_func=window_func)
    # if the tile is on an edge, don't window the edge side(s)
    if yn == 0:
        ywin[:pad_size * 2] = 1
    elif yn == ytot - 1:
        ywin[-pad_size * 2:] = 1
    if xn == 0:
        xwin[:pad_size * 2] = 1
    elif xn == xtot - 1:
        xwin[-pad_size * 2:] = 1
    # generate the 2D window
    win_2d = ywin.reshape(-1, 1).dot(xwin.reshape(1, -1))
    # return the windowed padded tile
    return np.pad(tile * win_2d, ((ybefore, yafter), (xbefore, xafter)),
                  mode='constant', constant_values=0)


def combine_img_with_padding(img_stack, pad_width):
    '''
    Reverse of split_img_with_padding
    '''
    assert pad_width % 2 == 0
    half_pad = pad_width // 2
    return combine_img(img_stack[..., half_pad:-half_pad, half_pad:-half_pad])


def split_process_recombine(fullpath, tile_size, padding, sim_kwargs,
                            extension='_split', bg_estimate=None,
                            window_func=cosine_edge):
    '''
    Method that splits then processes and then recombines images

    Returns
    -------
    total_save_path, sirecon_ouput
    '''
    # make sure zoomfact is integer
    zoom = sim_kwargs["zoomfact"]
    assert zoom.is_integer(), "Zoomfact is not an interger, Abort!"
    zoom = int(zoom)
    # open old Mrc
    oldmrc = Mrc.Mrc(fullpath)
    # pull data
    old_data = oldmrc.data
    # generate hex hash, will use as ID
    sha = hashlib.md5(old_data).hexdigest()
    sim_kwargs['sha'] = sha
    # split the data
    split_data = split_img_with_padding(old_data, tile_size, padding)
    num_tiles = split_data.shape[0]
    # estimate background
    if bg_estimate:
        bgs = {}
    # find local drive
    local_drive = os.path.expanduser('~')
    # make dir
    dir_name = os.path.join(local_drive, 'split_recon_' + sha)
    os.mkdir(dir_name)
    # save split data
    for i, data in tqdm.tqdm(
        enumerate(split_data), "Splitting and saving data", num_tiles, False):
        # save subimages in sub folder, use sha as ID
        savepath = os.path.join(dir_name,
                                'sub_image{:06d}_{}.mrc'.format(i, sha))
        Mrc.save(data, savepath, hdr=oldmrc.hdr, ifExists='overwrite')
        if bg_estimate == 'min':
            bgs[i] = data.min()
        elif bg_estimate == 'median':
            bgs[i] = np.median(data)
        elif bg_estimate == 'mode':
            bgs[i] = np.argmax(np.bincount(data.ravel()))
    # set up re
    i_re = re.compile('(?<=sub_image)\d+')
    # process data
    sirecon_ouput = []
    for path in tqdm.tqdm(glob.iglob(dir_name + '/sub_image*_{}.mrc'.format(sha)), "Processing tiles", num_tiles, False):
        # update the kwargs to have the input file.
        sim_kwargs.update({
            'input_file': path,
            'output_file': path.replace('.mrc', '_proc.mrc')
            })
        if bg_estimate:
            i = int(re.findall(i_re, path)[0])
            sim_kwargs['background'] = float(bgs[i])
        sirecon_ouput += simrecon(**sim_kwargs)
    # read in processed data
    recon_split_data = np.array([Mrc.Mrc(path).data[0]
                                 for path in sorted(glob.glob(dir_name +
                                        '/sub_image*_{}_proc.mrc'.format(sha)))
                                ])
    # recombine data, remember the data density is doubled so padding is to
    if window_func is None:
        recon_split_data_combine = combine_img_with_padding(recon_split_data,
                                                            padding * zoom)
    else:
        to_combine_data = None
        for i, d in tqdm.tqdm(enumerate(recon_split_data), "Recombining", num_tiles, False):
            current_tile = extend_and_window_tile(d, padding * zoom, i,
                                                  num_tiles,
                                                  window_func=window_func)
            if to_combine_data is None:
                to_combine_data = np.zeros_like(current_tile)
            to_combine_data += current_tile
        # still need to cut off the edges
        edge_pix = (padding // 2) * zoom
        slc = slice(edge_pix, -edge_pix, None)
        # cut them here.
        recon_split_data_combine = to_combine_data[slc, slc].astype(np.float32)
        # make sure the new data is the right shape
        oldy, oldx = old_data.shape[-2:]
        newy, newx = recon_split_data_combine.shape[-2:]
        assert newx == oldx * zoom, "X-dim: {} != {}".format(newx, oldx * zoom)
        assert newy == oldy * zoom, "Y-dim: {} != {}".format(newy, oldy * zoom)
    # save data
    temp_mrc = Mrc.Mrc(path.replace('.mrc', '_proc.mrc'))
    total_save_path = fullpath.replace(
        '.mrc', '_proc{}{}.mrc'.format(tile_size, extension))
    Mrc.save(recon_split_data_combine,
             total_save_path,
             hdr=temp_mrc.hdr,
             ifExists='overwrite')
    # clean up
    oldmrc.close()
    del oldmrc
    temp_mrc.close()
    del temp_mrc
    # kill folder
    shutil.rmtree(dir_name)

    return total_save_path, sirecon_ouput


def process_txt_output(txt_buffer):
    '''
    Take out put from above and parse into angles
    '''
    # compile the regexes
    ndir_re = re.compile('(?<=ndirs=)\d+', flags=re.M)
    angle_re = re.compile('(?:amplitude:\n In.*)(?<=angle=)(-?\d+\.\d+)',
                          flags=re.M)
    mag_re = re.compile('(?:amplitude:\n In.*)(?<=mag=)(-?\d+\.\d+)',
                        flags=re.M)
    amp_re = re.compile('(?:amplitude:\n In.*)(?<=amp=)(-?\d+\.\d+)',
                        flags=re.M)
    phase_re = re.compile('(?:amplitude:\n In.*)(?<=phase=)(-?\d+\.\d+)',
                          flags=re.M)
    # parse output
    my_dirs = set(re.findall(ndir_re, txt_buffer))
    assert len(my_dirs) == 1
    ndirs = int(list(my_dirs)[0])
    my_angles = np.array(re.findall(angle_re, txt_buffer)).astype(float)
    my_mags = np.array(re.findall(mag_re, txt_buffer)).astype(float)
    my_amps = np.array(re.findall(amp_re, txt_buffer)).astype(float)
    my_phases = np.array(re.findall(phase_re, txt_buffer)).astype(float)
    # find sizes
    assert len(my_angles) == len(my_mags) == len(my_amps) == len(my_phases)
    nx = ny = int(np.sqrt(len(my_angles) // ndirs))
    assert len(my_angles) == nx * ny * ndirs, '{} == {} * {} * {}'.format(len(my_angles), ndirs, nx, ny)
    # reshape all
    for data in (my_angles, my_mags, my_amps, my_phases):
        data.shape = (ny, nx, ndirs)
    # plot
    return my_angles, my_mags, my_amps, my_phases

def plot_params(angles, mags, amps, phases):
    titles = ('Angles', 'Magnitudes', 'Amplitudes', 'Phase')
    norients = angles.shape[-1]
    fig, axs = plt.subplots(4, norients, figsize=(norients * 4, 4 * 4))
    for row, data, t, c in zip(axs, (angles, mags, amps, phases),
                            titles, ('gnuplot2', 'gnuplot2', 'gnuplot2', 'seismic')):
        for i, ax in enumerate(row):
            # if angles we don't want absolute values.
            if 'Angles' in t:
                vmin = vmax = None
            else:
                vmin = data.min()
                vmax = data.max()
            # set the title for the middle row
            if i == 1:
                ax.set_title(t)
            ax.matshow(np.flipud(data[..., i]), vmin=vmin, vmax=vmax, cmap=c)
            ax.axis('off')
    fig.tight_layout()
    return fig, axs


def stitch_img(infile, labelfile, outfile, num_threads=1):
    '''
    Run the stitching algorithm

    More info can be found [here](https://github.com/mkazhdan/DMG)

    Parameters
    ----------
    infile : path
        Absolute path to the input file
    labelfile : path
        Absolute path to the label image
    outfile : path
        Absolute path to the output file
    num_threads : int (default, 1)
        The number of threads to run the problem across

    Returns
    -------
    client_return_code : string
        Output from client
    server_return_code : string
        Output from server
    '''
    # choose a communication port not in use
    COM_PORT = "12345"
    # set up client execution
    client_exc_str = [
        r"C:\DMG\ClientSocket.exe",     # binary location
        "--address", "127.0.0.1",       # address for client, points HOME
        "--port", COM_PORT,             # port to communicate over
        "--labels", labelfile,          # where's the label file, path
        "--pixels", infile,             # where's the input file, path
        "--threads", str(num_threads),  # How many threads to use for computation
        "--inCore",                     # tell algo to perform all computations in memory, __DO NOT__ stream to disk
        "--out", outfile,               # where do you want output
        "--hdr"                         # tell algorithm to use full 16 bit depth
    ]
    # set up server execution
    server_exc_str = [
        r"C:\DMG\ServerSocket.exe", # binary location
        "--count", "1",             # how many connections to expect
        "--port", COM_PORT,         # port to communicate over
        "--verbose",                # tell me what I'm doing
        "--tileExt", "tif",         # what file am I working with
        "--gray"                    # gray scale images, not color
    ]
    # both programs need to run concurrently so
    with subprocess.Popen(server_exc_str, stdout=subprocess.PIPE) as server:
        return_code = subprocess.check_output(client_exc_str)
        client_return_code = return_code.decode()
        server_return_code = server.stdout.read().decode()

    return client_return_code, server_return_code


def stitch_tiled_sim_img(sim_img_path, tile_size=None):
    '''
    A utility function to run the DMG stitching program on SIM data.
    '''
    # determin tile_size from file name if not given
    if tile_size is None:
        # make a regex
        tile_re = re.compile('(?<=proc)\d+')
        # apply it, there should only be one occurance
        re_result = re.findall(tile_re, sim_img_path)
        assert len(re_result) == 1, "More than one Regex found."
        tile_size = int(re_result[0])
    # prep the image to stitch
    to_stitch_path = sim_img_path.replace('.mrc', '.tif')
    # open the Mrc
    junk_mrc = Mrc.Mrc(sim_img_path)
    # pull data
    data = junk_mrc.data[0]
    # save tif version while filling up the bit depth
    tif.imsave(to_stitch_path, scale_uint16(data))
    labels = make_label_img(data.shape[-1] // 2, tile_size)
    # kill Mrc
    del junk_mrc
    head, tail = os.path.split(sim_img_path)
    label_file = head + os.path.sep + 'labels.tif'
    tif.imsave(label_file, labels)
    assert os.path.exists(label_file), label_file + " doesn't exist!"
    # prep outfile
    outfile = to_stitch_path.replace('.tif', '_stitch.tif')
    # stitch
    return_codes = stitch_img(to_stitch_path, label_file, outfile)
    return return_codes


def make_label_img(img_size, tile_size):
    # double tile size because SIM
    labels = np.array([np.ones((tile_size * 2, tile_size * 2), np.uint16) * i
                       for i in range((img_size // tile_size)**2)])
    return combine_img(labels)
