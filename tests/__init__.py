'''
Tests for random scripts
'''

import unittest
from nose.tools import *

class InitializationTests(unittest.TestCase):
    '''
    Initialization Tests for simrecon_utils
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
