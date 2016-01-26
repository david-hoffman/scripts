'''
# SIMRecon Utility Functions
'''
#import some os functionality so that we can be platform independent
import os
import numpy as np
#need subprocess to run commands
import subprocess
import pandas as pd

#import our ability to read MRC files
from pysegtools.mrc import MRC

#import skimage components
from skimage.external import tifffile as tif
from peaks.stackanalysis import PSFStackAnalyzer

from dphutils import slice_maker, fft_pad, Pupil, scale_uint16

class FakePSF(object):
    '''
    A class used to generate and save a FakePSF in MRC format using the `Pupil`
    class found in dphutils.
    '''

    def __init__(self, NA = 0.85, pixsize = 0.0975, det_wl = 520, n = 1.0):
        self.pupil = Pupil(1/(pixsize*1000), det_wl, NA, n)
        self.pixsize = pixsize
        #generate only the infocus

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

    def gen_psf(self, size = 512):
        self.pupil.size = size
        self.pupil.gen_psf([0])

        self.psf = scale_uint16(self.pupil.PSFi[0])

    def save_PSF_mrc(self, output_filename):
        '''
        Object specific wrapper for general save_PSF_mrc
        '''

        #take the best blob and pad to at least 512

        save_PSF_mrc(self.psf, output_filename, self.pixsize, self.det_wl)

class PSFFinder(object):

    def __init__(self, stack, psfwidth = 1.68, NA = 0.85, pixsize = 0.0975, det_wl = 520, window_width = 20, **kwargs):
        self.psfstackanalyzer = PSFStackAnalyzer(stack, psfwidth, **kwargs)
        self.all_blobs = self.psfstackanalyzer.peakfinder.blobs
        self.NA = NA
        self.pixsize = pixsize
        self.det_wl = det_wl
        self.window_width = window_width

        self.find_fit(2*psfwidth)
        self.find_best_psf()
        self.find_window()

    def find_fit(self, max_s = 2.1, num_peaks = 10):
        '''
        Function to find and fit blobs in the max intensity image

        Blobs with the appropriate parameters are saved for further fitting.

        Parameters
        ----------
        max_s : float
            Reject all peaks with a fit width greater than this
        num_peaks : int
            The number of peaks to analyze further
        '''
        window_width = self.window_width

        #pull the PeakFinder object
        my_PF = self.psfstackanalyzer.peakfinder
        #prune blobs
        my_PF.remove_edge_blobs(window_width)
        my_PF.prune_blobs(window_width)
        #fit blobs in max intensity
        blobs_df = my_PF.fit_blobs(window_width)

        blobs_df.SNR = np.round(blobs_df.dropna().SNR).astype(int)

        new_blobs_df = blobs_df[blobs_df.sigma_x < max_s].sort(['SNR','amp'],ascending=False).iloc[:num_peaks]
        #set the internal state to the selected blobs
        my_PF.blobs = new_blobs_df[['y0','x0','sigma_x','amp']].values.astype(int)

    def find_best_psf(self):
        '''
        Of the initial guesses found with find_fit() this finds the best
        '''
        my_PFSA = self.psfstackanalyzer
        my_PFSA.fitPeaks(self.window_width)

        my_PFSA.calc_psf_params()

        fits = my_PFSA.psf_params

        fits.SNR = np.round(fits.SNR).astype(int)
        #find min sigma_z peak
        best_fit = fits.sort(['SNR','sigma_z'],ascending=[False,True]).iloc[0]

        self.best_blob = best_fit

    def find_window(self):
        '''
        Finds the biggest window distance.
        '''

        #pull all blobs
        blobs = self.all_blobs
        best = np.round(self.best_blob[['y0', 'x0', 'sigma_x','amp']].values).astype(int)


        def calc_r(blob1, blob2):
            '''
            Calc euclidean distance between blob1 and blob2
            '''
            y1, x1, s1, a1 = blob1
            y2, x2, s2, a2 = blob2

            return np.sqrt((y1 - y2)**2 + (x1 - x2)**2)

        #calc distances
        r = np.array([calc_r(best, blob) for blob in blobs])

        #find min distances
        #remember that best is in blobs so 0 will be in the list
        #find the next value
        r.sort()
        try:
            r_min = r[1]
        except IndexError as e:
            #make r_min the size of the image
            r_min = min(np.array(self.psfstackanalyzer.stack.shape[1:3])-best[:2])

        #now window size equals sqrt or this
        win_size = int(round(2*(r_min/np.sqrt(2) - best[2]*3)))

        window = [self.best_blob.z0] + slice_maker(best[0], best[1], win_size)

        self.best_blob_data = self.psfstackanalyzer.stack[window]

    def save_PSF_mrc(self, output_filename):
        '''
        Object specific wrapper for general save_PSF_mrc
        '''

        #take the best blob and pad to at least 512
        img_raw = self.best_blob_data

        if img_raw.shape[0] < 512:
            img = fft_pad(img_raw, 512)
        else:
            img = fft_pad(img_raw)

        save_PSF_mrc(img,output_filename, self.pixsize, self.det_wl)

def save_PSF_mrc(img, output_filename, pixsize = 0.0975, det_wl = 520):
    '''
    A small utility function to save an image of a bead as an MRC

    Parameters
    ----------
    img : ndarray, rank 2
        The image to save
    output_filename : path
        The filename to output to
    pixsize : float
        the the pixel size in microns (size of the sensor pixel at the sample)
    det_wl : float
        the detection wavelength
    '''

    #TODO: make sure '.mrc' is appended to files that don't have it.

    ny, nx = img.shape
    PSFmrc = MRC(output_filename,nx=nx,ny=ny,dtype=img.dtype)
    PSFmrc.header['nz']=1
    PSFmrc[0] = img
    PSFmrc.header['nwave'] =1 #detection wavelength
    PSFmrc.header['wave1'] =det_wl #detection wavelength
    #need the rest of these fields filled out otherwise header won't write.
    PSFmrc.header['wave2'] =0
    PSFmrc.header['wave3'] =0
    PSFmrc.header['wave4'] =0
    PSFmrc.header['wave5'] =0
    #fill in the pixel size
    PSFmrc.header['xlen'] = pixsize
    PSFmrc.header['ylen'] = pixsize

    #need to delete this field to let MRC know that this is an oldstyle header to write
    del PSFmrc.header['cmap']

    #write the header and close the file.
    PSFmrc.write_header()
    PSFmrc.close()

    return output_filename

def calc_radial_mrc(infile, outfile = None, NA = 0.85, L = 8, H = 22):
    '''
    A simple wrapper around the radial OTF calc
    '''

    #TODO: Error checking

    if outfile is None:
        outfile = infile.replace('.mrc','_otf2d.mrc')

    #write our string to send to the shell
    #8 is the lower pixel and 22 is the higher pixel
    #0.8 is the detection NA
    otfcalc = r'C:\newradialft\otf2d -N {NA} -L {L} -H {H} {infile} {outfile}'

    #format the string
    # excstr = otfcalc.format(infile = infile,outfile = outfile, NA = NA, L = L, H = H)
    #send to shell
    # os.system(excstr)

    return_code = subprocess.call([r'C:\newradialft\otf2d', '-N', str(NA), '-L', str(L), '-H', str(H), infile, outfile])

    return return_code
