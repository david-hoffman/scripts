#! /usr/bin/env python
# -*- coding: utf-8 -*

import os
import glob
import textwrap
import numpy as np
import matplotlib as mpl
import multiprocessing as mp
from docopt import docopt
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
               **kwargs):
    '''
    Main function to generate and save thumbnail pngs
    '''
    # load data
    data = load_data(dirname, key)
    if data:
        # can clean the dirnames here
        data = {clean_dirname(k, figsize): v for k, v in data.items()}
        fig, ax = display_grid(data, figsize=figsize, **kwargs)
        foldername = os.path.abspath(dirname).split(os.path.sep)[-level]
        # save the figure.
        # fig.savefig(os.path.join(dirname, 'Thumbs ' + foldername + '.png'),
        #             bbox_inches='tight')
        if where == 'host':
            save_name = 'Thumbs ' + foldername + '.png'
        elif where == 'in folder':
            save_name = os.path.abspath(
                os.path.join(dirname, 'Thumbs ' + foldername + '.png'))
        else:
            save_name = os.path.abspath(
                os.path.join(where, 'Thumbs ' + foldername + '.png'))
        # make the layout 'tight'
        fig.tight_layout()
        # save the figure
        print('Saving', save_name, '...')
        fig.savefig(save_name, bbox_inches='tight')
        print('finished saving', save_name)
    # mark data for gc
    del data
    return dirname + os.path.sep + key


def gen_all_thumbs(home, path_key='SIM', **kwargs):
    '''
    '''
    with mp.Pool() as pool:
        results = [pool.apply_async(
                    gen_thumbs, args=(path,), kwds=kwargs
                ) for path in find_paths(home, path_key)]
        for pp in results:
            pp.get()
    # for path in find_paths(home, path_key):
    #     gen_thumbs(path, **kwargs)


if __name__ == '__main__':

    # grab the argsuments from the command line
    args = docopt(__doc__)

    # a little output so that the user knows whats going on
    print('Generating thumbnails', args['<myfile>'])

    # Need to take the first system argsument as the filename for a TIF file

    # test if filename has tiff in it
    filename = args['<myfile>']

    # try our main block
    try:
        if '.tif' in filename or '.tiff' in filename:
            # here's the real danger zone, did the user give us a real file?
            try:
                data = tif.imread(filename)
            except FileNotFoundError as er:
                raise er
            if args['--log']:
                import numpy as np
                if data.min() > 0:
                    data = np.log(data)
                else:
                    print(filename, 'had negative numbers, log not taken')

            # Trying to set the cmap here opens a new figure window
            # need to set up kwargss for efficient argsument passing
            # plt.set_cmap('gnuplot2')
            # plot the data
            fig, ax = mip(data)
            # readjust the white space (maybe move this into main code later)
            fig.subplots_adjust(top=0.85, hspace=0.3, wspace=0.3)
            # add an overall figure title that's the file name
            fig.suptitle(filename, fontsize=16)

            # check to see if we should make a PDF
            if args['--PDF']:
                fig.savefig(filename.replace('.tiff', '.pdf').replace('.tif', '.pdf'))
            else:
                # I still don't know why fig.show() doesn't work
                # I think it has something to do with pyplot's backend
                plt.show()
        else:
            # this is our own baby error handling, it avoids loading the
            # skimage package
            print('You didn\'t give me a TIFF')

    except Exception as er:
        print(er)
