'''
# SIMRecon Utility Functions
'''
#import some os functionality so that we can be platform independent
import os

#import our ability to read MRC files
from pysegtools.mrc import MRC

#import skimage components
from skimage.external import tifffile as tif
from peaks.stackanalysis import PSFStackAnalyzer

def save_PSF_mrc(img, output_filename, xlen = 0.0975, ylen = 0.0975, det_wl = 520):
    '''
    A small utility function to save an image of a bead as an MRC

    Parameters
    ----------

    '''

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
    PSFmrc.header['xlen'] = xlen
    PSFmrc.header['ylen'] = ylen

    #need to delete this field to let MRC know that this is an oldstyle header to write
    del PSFmrc.header['cmap']

    #write the header and close the file.
    PSFmrc.write_header()
    PSFmrc.close()

    return output_filename

def calc_radial_mrc(infile, outfile, NA = 0.85, L = 8, H = 22):
    '''
    A simple wrapper around the radial OTF calc
    '''

    #write our string to send to the shell
    #8 is the lower pixel and 22 is the higher pixel
    #0.8 is the detection NA
    otfcalc = r'C:\newradialft\otf2d -N {NA} -L {L} -H {H} {infile} {outfile}'

    #format the string
    excstr = otfcalc.format(infile = infile,outfile = outfile, NA = NA, L = L, H = H)

    #send to shell
    os.system(excstr)
