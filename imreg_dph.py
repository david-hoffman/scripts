#!/usr/bin/env python
# -*- coding: utf-8 -*-
# imreg_dph.py
"""
Functions for 2D image registration

Copyright (c) 2018, David Hoffman
"""
import itertools
import numpy as np
from dphutils import slice_maker

# three different registration packages
# not dft based
import cv2

# dft based
from skimage.feature import register_translation as register_translation_base
from skimage.transform import warp
from skimage.transform import AffineTransform as AffineTransformBase

try:
    import pyfftw

    pyfftw.interfaces.cache.enable()
    from pyfftw.interfaces.numpy_fft import fft2, ifft2, fftshift
except ImportError:
    from numpy.fft import fft2, ifft2, fftshift


class AffineTransform(AffineTransformBase):
    """Only adding matrix multiply to previous class"""

    def __matmul__(self, other):
        newmat = self.params @ other.params
        return AffineTransform(matrix=newmat)

    def __eq__(self, other):
        return np.array_equal(self.params, other.params)

    @property
    def inverse(self):
        return AffineTransform(matrix=np.linalg.inv(self.params))

    def __repr__(self):
        return self.params.__repr__()

    def __str__(self):
        string = (
            "<AffineTransform: translation = {}, rotation ={:.2f}," " scale = {}, shear = {:.2f}>"
        )
        return string.format(
            np.round(self.translation, 2),
            np.rad2deg(self.rotation),
            np.round(np.array(self.scale), 2),
            np.rad2deg(self.shear),
        )


AffineTransform.__init__.__doc__ = AffineTransformBase.__init__.__doc__
AffineTransform.__doc__ = AffineTransformBase.__doc__


def _calc_pad(oldnum, newnum):
    """ Calculate the proper padding for fft_pad

    We have three cases:
    old number even new number even
    >>> _calc_pad(10, 16)
    (3, 3)

    old number odd new number even
    >>> _calc_pad(11, 16)
    (2, 3)

    old number odd new number odd
    >>> _calc_pad(11, 17)
    (3, 3)

    old number even new number odd
    >>> _calc_pad(10, 17)
    (4, 3)

    same numbers
    >>> _calc_pad(17, 17)
    (0, 0)

    from larger to smaller.
    >>> _calc_pad(17, 10)
    (-4, -3)
    """
    # how much do we need to add?
    width = newnum - oldnum
    # calculate one side, smaller
    pad_s = width // 2
    # calculate the other, bigger
    pad_b = width - pad_s
    # if oldnum is odd and newnum is even
    # we want to pull things backward
    if oldnum % 2:
        pad1, pad2 = pad_s, pad_b
    else:
        pad1, pad2 = pad_b, pad_s
    return pad1, pad2


def highpass(shape):
    """Return highpass filter to be multiplied with fourier transform."""
    # inverse cosine filter.
    x = np.outer(
        np.cos(np.linspace(-np.pi / 2.0, np.pi / 2.0, shape[0])),
        np.cos(np.linspace(-np.pi / 2.0, np.pi / 2.0, shape[1])),
    )
    return (1.0 - x) * (2.0 - x)


def localize_peak(data):
    """
    Small utility function to localize a peak center. Assumes passed data has
    peak at center and that data.shape is odd and symmetric. Then fits a
    parabola through each line passing through the center. This is optimized
    for FFT data which has a non-circularly symmetric shaped peaks.
    """
    # make sure passed data is symmetric along all dimensions
    if not len(set(data.shape)) == 1:
        print("data.shape = {}".format(data.shape))
        return 0, 0
    # pull center location
    center = data.shape[0] // 2
    # generate the fitting lines
    my_pat_fft_suby = data[:, center]
    my_pat_fft_subx = data[center, :]
    # fit along lines, consider the center to be 0
    x = np.arange(data.shape[0]) - center
    xfit = np.polyfit(x, my_pat_fft_subx, 2)
    yfit = np.polyfit(x, my_pat_fft_suby, 2)
    # calculate center of each parabola
    x0 = -xfit[1] / (2 * xfit[0])
    y0 = -yfit[1] / (2 * yfit[0])
    # NOTE: comments below may be useful later.
    # save fits as poly functions
    # ypoly = np.poly1d(yfit)
    # xpoly = np.poly1d(xfit)
    # peak_value = ypoly(y0) / ypoly(0) * xpoly(x0)
    # #
    # assert np.isclose(peak_value,
    #                   xpoly(x0) / xpoly(0) * ypoly(y0))
    # return center
    return y0, x0


def logpolar(image, angles=None, radii=None):
    """Return log-polar transformed image and log base."""
    shape = image.shape
    center = shape[0] / 2, shape[1] / 2
    if angles is None:
        angles = shape[0]
    if radii is None:
        radii = shape[1]
    theta = np.empty((angles, radii), dtype=np.float64)
    theta.T[:] = -np.linspace(0, np.pi, angles, endpoint=False)
    # d = radii
    d = np.hypot(shape[0] - center[0], shape[1] - center[1])
    log_base = 10.0 ** (np.log10(d) / (radii))
    radius = np.empty_like(theta)
    radius[:] = np.power(log_base, np.arange(radii, dtype=np.float64)) - 1.0
    x = (radius / shape[1] * shape[0]) * np.sin(theta) + center[0]
    y = radius * np.cos(theta) + center[1]
    output = np.empty_like(x)
    ndii.map_coordinates(image, [x, y], output=output)
    return output, log_base


def translation(im0, im1):
    """Return translation vector to register images."""
    shape = im0.shape
    f0 = fft2(im0)
    f1 = fft2(im1)
    ir = fftshift(abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1)))))
    t0, t1 = np.unravel_index(np.argmax(ir), shape)
    dt0, dt1 = localize_peak(ir[slice_maker((t0, t1), 3)])
    # t0, t1 = t0 + dt0, t1 + dt1
    t0, t1 = np.array((t0, t1)) + np.array((dt0, dt1)) - np.array(shape) // 2
    # if t0 > shape[0] // 2:
    #     t0 -= shape[0]
    # if t1 > shape[1] // 2:
    #     t1 -= shape[1]
    return AffineTransform(translation=(-t1, -t0))


def similarity(im0, im1):
    """Return similarity transformed image im1 and transformation parameters.

    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector.

    A similarity transformation is an affine transformation with isotropic
    scale and without shear.

    Limitations:
    Image shapes must be equal and square.
    - can fix with padding, non-square images can be handled either with padding or
        better yet compensating for uneven image size
    All image areas must have same scale, rotation, and shift.
    - tiling if necessary...
    Scale change must be less than 1.8.
    - why?
    No subpixel precision.
    - fit peak position or upsample as in (https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/register_translation.py)

    """
    if im0.shape != im1.shape:
        raise ValueError("Images must have same shapes.")
    elif len(im0.shape) != 2:
        raise ValueError("Images must be 2 dimensional.")
    shape_ratio = im0.shape[0] / im0.shape[1]
    # calculate fourier images of inputs
    f0 = fftshift(abs(fft2(im0)))
    f1 = fftshift(abs(fft2(im1)))
    # high pass filter fourier images
    h = highpass(f0.shape)
    f0 *= h
    f1 *= h
    #     del h
    # convert images to logpolar coordinates.
    f0, log_base = logpolar(f0)
    f1, log_base = logpolar(f1)
    # fourier transform again ?
    f0 = fft2(f0)
    f1 = fft2(f1)
    # calculate impulse response
    r0 = abs(f0) * abs(f1)
    ir_cmplx = ifft2((f0 * f1.conjugate()) / r0)
    ir = abs(ir_cmplx)
    # find max, this fails to often and screws up when cluster processing.
    i0, i1 = np.unravel_index(np.argmax(ir), ir.shape)
    di0, di1 = localize_peak(ir[slice_maker((i0, i1), 3)])
    i0, i1 = i0 + di0, i1 + di1
    # calculate the angle
    angle = i0 / ir.shape[0]
    # and scale
    scale = log_base ** i1
    # if scale is too big, try complex conjugate of ir
    if scale > 1.8:
        ir = abs(ir_cmplx.conjugate())
        i0, i1 = np.array(np.unravel_index(np.argmax(ir), ir.shape))
        di0, di1 = localize_peak(ir[slice_maker((i0, i1), 5)])
        i0, i1 = i0 + di0, i1 + di1
        angle = -i0 / ir.shape[0]
        scale = 1.0 / (log_base ** i1)
        if scale > 1.8:
            raise ValueError("Images are not compatible. Scale change > 1.8")
    # center the angle
    angle *= np.pi
    if angle < -np.pi / 2:
        angle += np.pi
    elif angle > np.pi / 2:
        angle -= np.pi
    # apply scale and rotation
    # first move center to 0, 0
    # center shift is reversed because of definition of AffineTransform
    center_shift = np.array(im1.shape)[::-1] // 2
    af = AffineTransform(translation=center_shift)
    # then apply scale and rotation
    af @= AffineTransform(scale=(scale, scale), rotation=angle)
    # move back to center of image
    af @= AffineTransform(translation=-center_shift)
    # apply transformation
    im2 = warp(im1, af)
    # now calculate translation
    af @= translation(im0, im2)

    return af


def _convert_for_cv(im0):
    """Utility function to convert images to the right type."""
    # right now it doesn't do anything...
    if im0.dtype in {np.dtype("uint8"), np.dtype("float32")}:
        return im0
    return im0.astype("float32")


warp_dict = dict(
    homography=cv2.MOTION_HOMOGRAPHY,
    affine=cv2.MOTION_AFFINE,
    euclidean=cv2.MOTION_EUCLIDEAN,
    translation=cv2.MOTION_TRANSLATION,
)


def register_ECC(im0, im1, warp_mode="affine", num_iter=500, term_eps=1e-6):
    """Register im1 to im0 using findTransformECC from OpenCV

    Parameters
    ----------
    im0 : ndarray (2d)
        source image
    im1 : ndarray (2d)
        target image
    warp_mode : str
        type of warping
    num_iter : int
    term_eps : float
    """
    # make sure images are right type
    im0 = _convert_for_cv(im0)
    im1 = _convert_for_cv(im1)

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    warp_mode = warp_dict[warp_mode.lower()]
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iter, term_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    cc, warp_matrix = cv2.findTransformECC(im0, im1, warp_matrix, warp_mode, criteria)

    return AffineTransform(matrix=np.vstack((warp_matrix, (0, 0, 1))))


def register_translation(im0, im1, upsample_factor=100):
    """Right now this uses the numpy fft implementation, we can speed it up by
    dropping in fftw if we need to"""
    # run the skimage code
    shifts, error, phasediff = register_translation_base(im0, im1, upsample_factor)
    # the shifts it returns are negative and reversed of what we're expecting.
    af = AffineTransform(translation=-shifts[::-1])
    return af


def dual_registration(im0, im1, warp_mode=cv2.MOTION_AFFINE):
    af0 = similarity(im0, im1)
    im2 = warp(im1, af0).astype(im0.dtype)
    af1 = register_ECC(im0, im2, warp_mode=warp_mode)
    return af0 @ af1


def cv_warp(im, af, shape=None, **kwargs):
    """Driver function for cv2.warpAffine
    cv::INTER_NEAREST = 0,
    cv::INTER_LINEAR = 1,
    cv::INTER_CUBIC = 2,
    cv::INTER_AREA = 3,
    cv::INTER_LANCZOS4 = 4, """
    if shape is None:
        shape = im.shape
    default_kwargs = dict(flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    default_kwargs.update(kwargs)
    return cv2.warpAffine(im, af.params[:2], shape[::-1], **default_kwargs)


def propogate_transforms(transforms, normalize=False):
    """Propagate transforms along slabs"""
    # initialize
    initial = AffineTransform()
    # prepend a new transform
    transforms = [initial] + transforms
    # accumulate transforms
    new_transforms = list(itertools.accumulate(transforms, lambda a, b: b @ a))
    if normalize:
        middle_transform = new_transforms[len(new_transforms) // 2]
        new_transforms = [middle_transform.inverse @ transform for transform in new_transforms]
    return new_transforms


if __name__ == "__main__":
    import click
    import os
    import warnings
    import glob
    import tifffile as tif
    import dask
    from dask.diagnostics import ProgressBar

    @click.command()
    @click.option(
        "--directory",
        "-d",
        multiple=True,
        type=click.Path(exists=True, file_okay=False),
        help="Directory containing data",
    )
    @click.option(
        "--transform-type",
        "-t",
        default="translation",
        help="Transformation type. Available options are: \n-"
        + "\n-".join(sorted(warp_dict.keys())),
    )
    @click.option(
        "--combine-dirs",
        "-c",
        is_flag=True,
        help="If TRUE will treat multiple directories as single data set",
    )
    def cli(directory, transform_type, combine_dirs):
        """Register images within a folder

        Assumes all images can be ordered by glob and are single 2d images

        only works with tifs for now
        """
        if combine_dirs:
            click.echo("Treating {} as a single data set".format(directory))
        # iterate through directories
        all_images = []
        all_names = []

        for d in directory:
            click.echo("Registering *.tif images in {}".format(os.path.abspath(d)))
            # add trailing slash
            d = os.path.join(d, "")
            resultname = os.path.dirname(d)
            click.echo("Loading images ...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with ProgressBar():
                    images = dask.delayed(
                        [dask.delayed(tif.imread)(path) for path in sorted(glob.glob(d + "*.tif"))]
                    ).compute()
            if not combine_dirs:
                register_save(images, transform_type, resultname)
            else:
                all_images += images
                all_names.append(resultname)

        if combine_dirs:
            click.echo()
            register_save(all_images, transform_type, " ".join(all_names))

    def register_save(images, transform_type, resultname):
        click.echo("Computing transforms ...")
        with ProgressBar():
            transforms = dask.delayed(
                [
                    dask.delayed(register_ECC)(images[i], images[i + 1], transform_type)
                    for i in range(len(images) - 1)
                ]
            ).compute()

        click.echo("Warping images ...")
        with ProgressBar():
            images_reg = dask.delayed(
                [
                    dask.delayed(cv_warp)(im, af)
                    for im, af in zip(images, propogate_transforms(transforms, normalize=True))
                ]
            ).compute()

        resultname += ".tif"
        click.echo("Saving results in {}".format(resultname))
        tif.imsave(resultname, np.asarray(images_reg))

    cli()
