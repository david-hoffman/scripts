'''
Tests for random scripts
'''

import unittest
from nose.tools import *
from palm_diagnostics import measure_peak_widths
import itertools as itt
import numpy as np


class InitializationTests(unittest.TestCase):
    '''
    Initialization Tests for simrecon_utils
    '''

    def test_initialization(self):
        """
        Check the test suite runs by affirming 2+2=4
        """
        self.assertEqual(2 + 2, 4)

    def test_import(self):
        """
        Ensure the test suite can import our module
        """
        try:
            import simrecon_utils as su
        except ImportError:
            self.fail("Was not able to import the simrecon_utils package")


def _test_trace(trace):
    """Sub test for making sure our peak_width measurement matches what it should"""
    peak_width_sum = measure_peak_widths(trace).sum()
    total_nonzero = np.nonzero(trace)[0].size
    assert peak_width_sum == total_nonzero, "{} != {}, Trace is {}".format(peak_width_sum, total_nonzero, trace)


def test_measure_peak_widths():
    """Test well definited cases"""
    for i in range(2, 6):
        for trace in itt.product((0, 1), repeat=i):
            _test_trace(trace)


def test_measure_peak_widths_random():
    """Test a random case"""
    size = np.random.randint(50, 10000)
    p = np.random.rand(1)
    trace = np.random.binomial(1, p, size)
    _test_trace(trace)