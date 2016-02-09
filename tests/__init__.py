'''
Tests for random scripts
'''

import unittest
from nose.tools import *
import numpy as np

class InitializationTests(unittest.TestCase):
    '''
    Initialization Tests
    '''

    def test_initialization(self):
        """
        Check the test suite runs by affirming 2+2=4
        """
        self.assertEqual(2+2, 4)

    def test_import(self):
        """
        Ensure the test suite can import our module
        """
        try:
            import simrecon_utils as su
        except ImportError:
            self.fail("Was not able to import the simrecon_utils package")

import simrecon_utils as su

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
