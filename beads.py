#!/usr/bin/env python
# -*- coding: utf-8 -*-
# beads.py
"""
Calculate bead densities.

Copyright (c) 2016, David Hoffman
"""

import click
from scipy.constants import Avogadro, pi

def good_N(ci,M, Nf = 5e10):
    '''
    This function is for converting a dye solution into desired number of particles.
    
    Parameters
    ----------
    ci : float
        initial concentration in mg/mL
    M : float
        molar mass (g/mol)
    Nf : float
        the number of particles per mL in the final dilution
    
    Returns
    -------
    Vf : float
        The final dilution
        
    Example
    -------
    >>> print('{:.3e}'.format(good_N(631/8*1e5))
    7.888e+06
    '''
    return ci/(M*1e3)/Nf*Avogadro
    
def num_particles(C, diameter, rho = 1.05):
    '''
    Equation from https://tools.thermofisher.com/content/sfs/manuals/mp05000.pdf
    
    Parameters
    ----------
    C : float
        Concentration of solution in g/mL
    diameter : float
        diameter of the beads in microns
    rho : float
        density of beads in g/mL (default: 1.05 for polystyrene)
        
    Returns
    -------
    N : float
        number of microspheres/mL
        
    Example
    -------
    >>> print('{:.3e}'.format(num_particles(0.02, 10)))
    3.638e+7
    '''
    return 6 * C * 1e12 / (rho * pi * diameter**3)

def molarity(num_part):
    """Convert number of particles per mL to molarity""" 
    return num_part * 1000 / Avogadro / 1e-12


@click.command()
@click.option('--count', default=1, help='number of greetings')
@click.argument('name')
def main():


if __name__ == '__main__':
    main()