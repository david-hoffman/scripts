import unittest
import os
from nose.tools import *
import numpy as np
import simrecon_utils as su
from Mrc import Mrc


def test_split_img_args():
    '''
    Testing that split_img rejects arguments without the right shape
    '''
    fake_data = np.random.randn(1023, 1023)
    assert_raises(AssertionError, su.split_img, fake_data, 4)

    fake_data = np.random.randn(1024, 1024)
    assert_raises(AssertionError, su.split_img, fake_data, 9)


def test_img_split_combine():
    '''
    Testing that the split and combine functions are opposites of one another
    '''

    fake_data = np.random.randn(1024, 1024)

    split_data = su.split_img(fake_data, 4)

    combine_data = su.combine_img(split_data[:, 0])

    assert np.all(fake_data == combine_data)


def test_img_split_combine2():
    '''
    Testing that the split and combine functions work properly with real data
    '''

    data_path = os.path.join('fixtures', '488 nm SIM 0.80_cam1_0.mrc')
    data = Mrc(data_path).data

    # split data 4 ways and then take mean along second dimension
    # (phase dimension)
    split_data = su.split_img(data, 4).mean(1)
    # combine data
    combine_data = su.combine_img(split_data)
    # compare to straight mean.
    assert np.all(data.mean(0) == combine_data)
