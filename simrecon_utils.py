'''
SIMRecon Utility Functions
'''
# import some os functionality so that we can be platform independent
import os
import numpy as np
import matplotlib.pyplot as plt
# need subprocess to run commands
import subprocess

# import our ability to read and write MRC files
import Mrc

from collections import OrderedDict, Sequence
# import skimage components
from peaks.peakfinder import PeakFinder

from dphutils import (slice_maker, Pupil, scale_uint16, fft_pad,
                      nextpow2, radial_profile)
from pyfftw.interfaces.numpy_fft import ifftshift, fftshift, fftn, ifftn


class FakePSF(object):
    '''
    A class used to generate and save a FakePSF in MRC format using the `Pupil`
    class found in dphutils.
    '''

    def __init__(self, NA=0.85, pixsize=0.0975, det_wl=520, n=1.0):
        self.pupil = Pupil(1/(pixsize*1000), det_wl, NA, n)
        self.pixsize = pixsize
        # generate only the infocus

    def NA():
        doc = "The NA property."

        def fget(self):
            return self.pupil.NA

        def fset(self, value):
            self.pupil.NA = value

        def fdel(self):
            del self.pupil.NA
        return locals()
    NA = property(**NA())

    def det_wl():
        doc = "The det_wl property."

        def fget(self):
            return self.pupil.wl

        def fset(self, value):
            self.pupil.wl = value

        def fdel(self):
            del self.pupil.wl
        return locals()
    det_wl = property(**det_wl())

    # def pixsize():
    #     doc = "The pixsize property."
    #     def fget(self):
    #         return self.pixsize
    #     def fset(self, value):
    #         self.pixsize = value
    #         self.pupil.k_max = 1/(value*1000)
    #     def fdel(self):
    #         del self.pupil.pixsize
    #         del self.pupil.k_max
    #     return locals()
    # pixsize = property(**pixsize())

    def gen_psf(self, size=512):
        self.pupil.size = size
        self.pupil.gen_psf([0])

        self.psf = scale_uint16(self.pupil.PSFi[0])

    def gen_radialOTF(self):
        psf = self.pupil.PSFi[0].copy()

        # pull the max size
        nx = max(psf.shape)

        # calculate the um^-1/pix
        dkr = 1/(nx*self.pixsize)
        # save dkr for later
        self.dkr = dkr
        # calculate the kspace cutoff, round up (that's what the 0.5 is for)
        krcutoff = int(2*self.NA/(self.det_wl/1000)/dkr + .5)

        self.radprof = calc_radial_OTF(psf, krcutoff)

    def save_radOTF_mrc(self, output_filename, **kwargs):
        # make empty header
        header = Mrc.makeHdrArray()
        # initialize it
        # set type and shape
        Mrc.init_simple(header, 4, self.radprof.shape)
        # set wavelength
        header.wave = self.det_wl
        # set number of wavelengths
        header.NumWaves = 1
        # set dimensions
        header.d = (self.dkr,)*3
        tosave = self.radprof.astype(np.complex64)
        # save it
        tosave = tosave.reshape(1, 1, len(tosave))

        Mrc.save(tosave, output_filename, hdr=header, **kwargs)

    def save_PSF_mrc(self, output_filename):
        '''
        Object specific wrapper for general save_PSF_mrc
        '''

        # take the best blob and pad to at least 512

        save_PSF_mrc(self.psf, output_filename, self.pixsize, self.det_wl)


class PSFFinder(object):

    def __init__(self, stack, psfwidth=1.68, NA=0.85, pixsize=0.0975,
                 det_wl=520, window_width=20, **kwargs):
        self.stack = stack
        self.peakfinder = PeakFinder(stack.max(0), psfwidth, **kwargs)
        self.peakfinder.find_blobs()
        self.all_blobs = self.peakfinder.blobs
        self.NA = NA
        self.pixsize = pixsize
        self.det_wl = det_wl
        self.window_width = window_width

        self.find_fit(2*psfwidth)

    def find_fit(self, max_s=2.1, num_peaks=20):
        '''
        Function to find and fit blobs in the max intensity image

        Blobs with the appropriate parameters are saved for further fitting.

        Parameters
        ----------
        max_s: float
            Reject all peaks with a fit width greater than this
        num_peaks: int
            The number of peaks to analyze further
        '''
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
        blobs_df.SNR = np.round(blobs_df.dropna().SNR).astype(int)
        # sort by SNR then sigma_x.
        new_blobs_df = blobs_df[
                        blobs_df.sigma_x < max_s
                ].sort(['SNR', 'sigma_x'], ascending=False).iloc[:num_peaks]
        # set the internal state to the selected blobs
        my_PF.blobs = new_blobs_df[
                                    ['y0', 'x0', 'sigma_x', 'amp']
                                ].values.astype(int)

        self.fits = new_blobs_df

    def find_window(self, blob_num=0):
        '''
        Finds the biggest window distance.
        '''

        # pull all blobs
        blobs = self.all_blobs
        # three different cases
        if not len(blobs):
            # no blobs in window, raise hell
            raise RuntimeError("No blobs found, can't find window")
        else:
            # more than one blob find
            best = np.round(
                self.fits.iloc[blob_num][['y0', 'x0', 'sigma_x', 'amp']].values
                ).astype(int)

            def calc_r(blob1, blob2):
                '''
                Calc euclidean distance between blob1 and blob2
                '''
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
                    np.concatenate((np.array(self.stack.shape[1:3])-best[:2],
                                    best[:2]))
                    )

            # now window size equals sqrt or this
            win_size = int(round(2*(r_min/np.sqrt(2) - best[2]*3)))

        window = slice_maker(best[0], best[1], win_size)
        self.window = window

        return window

    def plot_window(self, blob_num, **kwargs):
        '''
        Plot all the things for this window
        '''

        self.find_window(blob_num)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.matshow(self.peakfinder.data[self.window])
        self.gen_radialOTF(show_OTF=True, **kwargs)
        ax2.semilogy(abs(self.radprof))
        fig.tight_layout()

    def calc_infocus_psf(self):
        '''
        Calculate the infocus psf
        '''
        img_raw = self.stack[
            [slice(None, None, None)] + self.window
        ]

        img = fft_pad(img_raw)
        # estimate the background, use the mode.
        try:
            offset = np.bincount(img_raw.ravel()).argmax()
        except TypeError:
            offset = np.median(img_raw)
        # remove background from PSF
        psf = img.astype(float)-offset
        # recenter
        # TODO: add this part

        # fft
        otf = ifftshift(fftn(fftshift(psf)))
        # ifft
        self.psf = abs(ifftshift(ifftn(fftshift(otf.mean(0)))))

    def gen_radialOTF(self, lf_cutoff=0.1, width=3, **kwargs):
        '''
        Generate the Radially averaged OTF from the sample data.
        '''

        img_raw = self.stack[
            [slice(None, None, None)] + self.window
        ]

        if img_raw.shape[-1] < 512 or img_raw.shape[-2] < 512:
            img = fft_pad(img_raw, (nextpow2(img_raw.shape[0]), 512, 512))
        else:
            img = fft_pad(img_raw)

        # pull the max x, y size
        nx = max(img.shape[-1:-3:-1])

        # calculate the um^-1/pix
        dkr = 1/(nx*self.pixsize)
        # save dkr for later
        self.dkr = dkr
        # calculate the kspace cutoff, round up (that's what the 0.5 is for)
        krcutoff = int(2*self.NA/(self.det_wl/1000)/dkr + .5)

        radprof = calc_radial_OTF(img, krcutoff, **kwargs)

        if lf_cutoff is not None:
            # if given a cutoff linearly fit the points around it to a line
            # then interpolate the line back to the origin starting at the low
            # frequency

            # find the cutoff in terms of pixels
            mid_num = int(lf_cutoff/dkr + .5)

            # choose the low frequency
            lf = mid_num-width
            if lf > 0:
                # if the low frequency is higher than the DC component then
                # proceed with the fit, we definitely don't want to include
                # the DC
                hf = mid_num+width
                m, b = np.polyfit(np.arange(lf, hf), radprof[lf:hf], 1)

                radprof[:mid_num] = np.arange(0, mid_num)*m + b

            else:
                # set DC to mid_num to mid_num
                radprof[:mid_num] = radprof[mid_num]

            radprof /= radprof.max()

        self.radprof = radprof

        print('Better cutoff is {:.3f}'.format(
            (self.radprof[:krcutoff].argmin() -
             1)/(2/(self.det_wl/1000)/self.dkr)))

    def save_radOTF_mrc(self, output_filename, **kwargs):
        # make empty header
        header = Mrc.makeHdrArray()
        # initialize it
        # set type and shape
        Mrc.init_simple(header, 4, self.radprof.shape)
        # set wavelength
        header.wave = self.det_wl
        # set number of wavelengths
        header.NumWaves = 1
        # set dimensions
        header.d = (self.dkr,)*3
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


def calc_radial_mrc(infile, outfile=None, NA=0.85, L=8, H=22):
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
    # 0.8 is the detection NA
    otfcalc = r'C:\newradialft\otf2d -N {NA} -L {L} -H {H} {infile} {outfile}'

    # format the string
    # excstr = otfcalc.format(infile=infile, outfile=outfile, NA=NA, L=L, H=H)
    # send to shell
    # os.system(excstr)

    return_code = subprocess.call([
            r'C:\newradialft\otf2d',
            '-N', str(NA),
            '-L', str(L),
            '-H', str(H),
            infile,
            outfile
        ])

    return return_code


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
        data acquired with fast live SIM
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
        the (effective) NA of the objective
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
        'forcemodamp',
        'nok0search',
        'nokz0',
        'k0searchAll',
        'k0angles',
        'fitonephase',
        'noapodizeout',
        'gammaApo',
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
        'nordersout',
        'angle0',
        'negDangle',
        'ls',
        'na',
        'nimm',
        'saveprefiltered',
        'savealignedraw',
        'saveoverlaps'
    ))

    valid_kwargs.update({
        'ndirs': int,
        'nphases': int,
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
        'zoomfact': float,
        'zzoom': float,
        'zpadto': int,
        'explodefact': float,
        'nofilteroverlaps': bool,
        'nosuppress': bool,
        'suppressR': float,
        'dampenOrder0': bool,
        'noOrder0': bool,
        'noequalize': bool,
        'equalizez': bool,
        'equalizet': bool,
        'wiener': float,
        'wienerInr': float,
        'background': float,
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
                assert os.path.exists(kw_value), '{} is an invalid path'.format(kw_value)
            else:
                assert isinstance(kw_value, kw_type), '{} is type {} and should have been type {}'.format(k, type(kw_value), repr(kw_type))
            if kw_type is bool:
                if kw_value:
                    exc_list.append('-'+k)
            else:
                exc_list.append('-'+k)
                if isinstance(kw_value, Sequence):
                    for k in kw_value:
                        exc_list.append(str(k))
                else:
                    exc_list.append(str(kw_value))

    # save the output
    return_code = subprocess.check_output(exc_list)

    return return_code.decode('utf-8').split('\n')


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
    # need to add bit to move max to center.
    # estimate the background, use the mode.
    try:
        offset = np.bincount(psf.ravel()).argmax()
    except TypeError:
        offset = np.median(psf)
    # remove background from PSF
    newpsf = psf.astype(float)-offset
    # recenter
    # TODO: add this part

    # fft
    otf = ifftshift(fftn(fftshift(newpsf)))

    if show_OTF:
        from dphplotting.mip import mip
        from matplotlib.colors import LogNorm
        # this is still wrong, need to do the mean before summing
        # really we need a slice function.
        mip(abs(otf), func=np.mean, norm=LogNorm())

    if otf.ndim > 2:
        # if we have a 3D OTF collapse it by summation along kz into a 2D OTF.
        otf = otf.mean(0)

    center = np.array(otf.shape)/2

    radprof = (radial_profile(np.real(otf), center) +
               radial_profile(np.imag(otf), center)*1j)[:int(center[0]+1)]

    radprof /= radprof.max()

    if krcutoff is not None:
        # set everything beyond the diffraction limit to 0
        radprof[krcutoff:] = 0

    return radprof


def split_img(img, side):
    '''
    A utility to split a SIM stack into substacks
    '''

    # Testing input
    divisor = img.shape[-1]//side
    # Error checking
    assert side == img.shape[-1]/divisor, 'Side {}, not equal to {}/{}'.format(
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
                       ).reshape(ylen*divisor, xlen*divisor)


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
        window = [slice(None, None, None)] + slice_maker(ny//2, nx//2,
                                                         max(ny, nx)//2)
    # prepare a new file to write to
    Mrc.save(old_data[window], croppath, ifExists='overwrite', hdr=oldmrc.hdr)
    # close the old MRC file.
    oldmrc.close()
    del oldmrc
    return croppath

