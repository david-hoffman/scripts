#! /usr/bin/env python
# -*- coding: utf-8 -*

import os
import glob
import textwrap
import numpy as np
import matplotlib as mpl
import multiprocessing as mp
from skimage.external import tifffile as tif
from dphplotting import display_grid

# TODO: we could make a subclass of dict that would only contain
# keys. But when __getitem__ was called it would load the data
# associated with value. This way values would be available for
# gc once they've been used


def find_paths(home='.', key='SIM'):
    '''
    Utility function to return path names of folders
    whose name match key, starting at home
    '''
    # walk the directory starting a root
    for dirpath, dirnames, fnames in os.walk(home):
        # if SIM in the path then kill root and return directory
        if key in dirpath:
            dirnames.clear()
            yield dirpath


def load_data(dirname, key):
    '''
    Function to load all specified data matching glob into a dict
    and return the dict
    '''

    def better_imread(fname):
        '''
        Small utility to side step OME parsing error.
        '''
        with tif.TiffFile(fname) as mytif:
            return np.squeeze(np.array([page.asarray()
                                        for page in mytif.pages]))

    return {fname: better_imread(fname)
            for fname in glob.iglob(dirname + os.path.sep + key)
            if '.tif' in fname}


def clean_dirname(dirname, figsize):
    '''
    We only want the last two entries of dirname

    Filename and folder name

    and we want to clean those as well
    '''
    path = dirname.split(os.path.sep)
    fontsize = mpl.rcParams['font.size']
    # 1/120 = inches/(fontsize*character)
    num_chars = int(figsize / fontsize * 100)
    foldername = textwrap.fill(path[-2], num_chars)
    filename = textwrap.fill(path[-1], num_chars)
    return foldername + '\n' + filename


def gen_thumbs(dirname, key='/*/*decon.tif', where='host', level=2, figsize=6,
               redo=True, **kwargs):
    '''
    Main function to generate and save thumbnail pngs
    '''
    # load data
    data = load_data(dirname, key)
    if data:
        # can clean the dirnames here
        foldername = os.path.abspath(dirname).split(os.path.sep)[-level]
        if where == 'host':
            save_name = 'Thumbs ' + foldername + '.png'
        elif where == 'in folder':
            save_name = os.path.abspath(
                os.path.join(dirname, 'Thumbs ' + foldername + '.png'))
        else:
            save_name = os.path.abspath(
                os.path.join(where, 'Thumbs ' + foldername + '.png'))
        if not redo and os.path.exists(save_name):
            print(save_name, "already exists, skipping")
            return dirname + os.path.sep + key
        data = {clean_dirname(k, figsize): v for k, v in data.items()}
        fig, ax = display_grid(data, figsize=figsize, **kwargs)
        # make the layout 'tight'
        fig.tight_layout()
        # save the figure
        print('Saving', save_name, 'on', os.getpid(), '...')
        fig.savefig(save_name, bbox_inches='tight')
        print('finished saving', save_name)
    # mark data for gc
    del data
    return dirname + os.path.sep + key


def gen_all_thumbs(home, path_key='SIM', **kwargs):
    '''
    '''
    with mp.Pool() as pool:
        # spread jobs over processors.
        results = [pool.apply_async(
            gen_thumbs, args=(path,), kwds=kwargs
        ) for path in find_paths(home, path_key)]
        for pp in results:
            # workers take care of output so nothing needs to be saved
            pp.get()


if __name__ == '__main__':
    raise NotImplementedError
    # For cmd line utility someday.
