#!/usr/bin/env python
# -*- coding: utf-8 -*-
# beads.py
"""
Calculate bead densities.

Copyright (c) 2016, David Hoffman
"""

import click
from scipy.constants import Avogadro, pi


def good_N(ci, M, Nf=5e10):
    """
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
    """
    return ci / (M * 1e3) / Nf * Avogadro


def num_particles(concentration, diameter, rho=1.05):
    """
    Equation from https://tools.thermofisher.com/content/sfs/manuals/mp05000.pdf
    
    Parameters
    ----------
    concentration : float
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
    """
    return 6 * concentration * 1e12 / (rho * pi * diameter ** 3)


def calc_molarity(num_part):
    """Convert number of particles per mL to pico molarity"""
    return num_part * 1000 / Avogadro / 1e-12


@click.group("name")
def main():
    """Parameters
    ----------
    concentration : float
        Concentration of solution in g/mL
    diameter : float
        diameter of the beads in microns
    rho : float
        density of beads in g/mL (default: 1.05 for polystyrene)"""
    pass


def sub_molarity(diameter, concentration, rho):
    """Calculate the molarity of solution"""
    num_part = num_particles(concentration, diameter, rho=rho)
    mol = calc_molarity(num_part)
    click.echo(
        (
            "For a bead diameter of {} um and a concentration of {} "
            "g/mL you have a {:.3f} pM solution"
        ).format(diameter, concentration, mol)
    )
    return mol


@main.command()
@click.argument("diameter", type=float)
@click.argument("concentration", type=float)
@click.option("--rho", default=1.05, type=float)
def molarity(diameter, concentration, rho):
    sub_molarity(diameter, concentration, rho)


@main.command()
@click.argument("diameter", type=float)
@click.argument("concentration", type=float)
@click.argument("desired", type=float)
@click.option("--rho", default=1.05, type=float)
def dilution(diameter, concentration, desired, rho):
    """Calculate the molarity of solution everything in pM"""
    mol = sub_molarity(diameter, concentration, rho=rho)
    click.echo(
        "To get a concentration of {} pM you'd need to do a 1:{:.0f} dilution".format(
            desired, mol / desired
        )
    )


molarity.__doc__ = num_particles.__doc__

if __name__ == "__main__":
    main()
