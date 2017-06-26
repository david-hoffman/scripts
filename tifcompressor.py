#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tifcompressor.py
"""
CLI for compressing tifs

Copyright (c) 2017, David Hoffman
"""

import click
import os
import glob
from skimage.external import tifffile as tif
import dask
import dask.distributed
# out = c.compute(collection)   # c is the client
# progress(out)

# register dask progress bar
# ProgressBar().register()


@dask.delayed(pure=True)
def compress(path, compression):
    """Compress a tif file, requires that the tif can be read into memory"""
    with tif.TiffFile(path) as img:
        if img.pages[0].compression != "deflate":
            data = img.asarray()
            tif.imsave(path, data, compress=compression)


@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=False,
                                       readable=True, allow_dash=False))
@click.option("--compression", "-c", default=6, type=click.IntRange(0, 9),
              help=("Values from 0 to 9 controlling the level of zlib compression."
              "If 0, data are written uncompressed (default). Compression cannot be"
              "used to write contiguous files. If 'lzma', LZMA compression is used,"
              "which is not available on all platforms."))
@click.option("--recursive", "-r", is_flag=True,
              help="Recursively search SRC for tifs")
def cli(src, compression, recursive):
    """Compress the tif files in dir by compression amount.
    """
    # get a sorted list of all the files
    globpat = "/*.tif"
    if recursive:
        globpat = "/**" + globpat

    click.echo("Starting local cluster")
    cluster = dask.distributed.LocalCluster()
    client = dask.distributed.Client(cluster)

    click.echo("Searching for files in {} ... ".format(os.path.abspath(src) + globpat))
    # Need to make this asynchronous (start farming out work as soon as possible.)
    futures = [client.submit(compress, path, compression) for path in glob.iglob(src + globpat, recursive=recursive)]
    # click.echo("found {} files".format(len(to_compute)))
    # track the computation.
    dask.distributed.progress(futures)


if __name__ == "__main__":
    cli()
