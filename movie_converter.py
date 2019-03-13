#!/usr/bin/env python
# -*- coding: utf-8 -*-
# movie_converter.py
"""
Convert from shitty amira format to clean format

Copyright David Hoffman, 2018
"""

import cv2
import glob
import dask
import os

new_ext = "_conv.mpg"

def update_path(path):
    return path.replace(".mpg", new_ext)

@dask.delayed
def convert_file(path):
    """Convert movie file with opencv"""
    print(path)
    # open the video file
    cap = cv2.VideoCapture(path)
    cap.retrieve()
    # set the codec (only one that works here)
    fourcc = cv2.VideoWriter_fourcc(*'M1V1')

    # begin loop
    out = None
    while True:
        # try and get next frame
        ret, frame = cap.read()
        if not ret:
            break
        # initialize for first iteration
        if out is None:
            # writer expects (width, height) tuple for shape
            out =  cv2.VideoWriter(update_path(path), fourcc, 25.0, frame.shape[:2][::-1], True)
        # write frame
        out.write(frame)

    # close objects
    cap.release()
    out.release()
    return path

def new_files():
    # filter out converted files
    paths = filter(lambda x: new_ext not in x, glob.iglob("*.mpg"))
    for path in paths:
        t_orig = os.path.getmtime(path)
        try:
            t_conv = os.path.getmtime(update_path(path))
            if t_orig > t_conv:
                # converted file is older then original file
                yield path
        except FileNotFoundError:
            # no converted file
            yield path


def main():
    # convert all files in the folder
    print(dask.delayed(list(map(convert_file, new_files()))).compute(scheduler="processes"))

if __name__ == '__main__':
    main()