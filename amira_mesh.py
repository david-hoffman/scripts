#!/usr/bin/env python
# -*- coding: utf-8 -*-
# decay_utils.py
"""
Utilities to read (not implemented) and write Amira Mesh files

Only writing point files is implemented for now.

Copyright (c) 2018, David Hoffman
"""

import os
import numpy as np

header = """# AmiraMesh BINARY-LITTLE-ENDIAN 2.1

# number of points in the file
define Points {num_points:d}
# number of label_labels ... ?
define LABEL_Labels {num_labels:d}

Parameters {{
    _symbols {{
            {symbols:s}
        }}
    ContentType "HxCluster"
}}

# Points have 3 coordinates, store as var 1
Points {{ float[3] Coordinates }} @1
# one id
Points {{ int Ids }} @2
# data columns
{columns:s}
# three labels
{labels:s}

# Data section follows
# Points {{ float[3] Coordinates }} @1, no delimiter, know it's in groups of three from header
"""

column_base = "Points {{ float Data }} @{:d}"
label_base = "LABEL_Labels {{ byte strings }} @{:d}"


def _normalize_str(s):
    """Does nothing now, will remove invalid characters in the future"""
    return s


def write_and_pack(num, data, file):
    """Take data, make sure it's flat, write to file with var identifier"""
    assert data.ndim == 1, "data wrong shape"
    # < signifies little endian
    if np.issubdtype(data.dtype, np.inexact):
        data = data.astype(np.float32)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.int32)
    packed = data.tobytes()
    var = "\n@{}\n".format(num)
    file.writelines([var.encode(), packed])


def export_mesh(fname, dataframe, xyz_col=["x0", "y0", "z0"], label_cols=[], id_col=[]):
    """Export Amira mesh file representing a point cloud (HxCluster)"""
    if os.path.splitext(fname)[-1] != ".am":
        fname = fname + ".am"

    assert 0 <= len(id_col) <= 1, "id_col not the right length"
    data_col = dataframe.columns.difference(xyz_col + label_cols + id_col)
    # start putting things together that we know
    format_dict = dict(
        num_points=len(dataframe),
        num_labels=len(label_cols),
        # normalize symbols
        symbols="\n,".join(["C{:03d} ".format(i) + _normalize_str(s) for i, s in enumerate(data_col)])
    )

    start_of_data = 3
    end_of_data = start_of_labels = start_of_data + len(data_col)
    format_dict["columns"] = "\n".join([column_base.format(i) for i in range(start_of_data, end_of_data)])

    end_of_labels = end_of_data + format_dict["num_labels"]
    format_dict["labels"] = "\n".join([label_base.format(i) for i in range(start_of_labels, end_of_labels)])

    with open(fname, "w") as file:
        file.write(header.format(**format_dict))

    with open(fname, "ab") as file:
        # convert header to bytes and write to file
        # write data
        xyz = dataframe[xyz_col].values.ravel()
        write_and_pack(1, xyz, file)

        # write index
        if len(id_col):
            idx = dataframe[id_col]
        else:
            idx = dataframe.index.values
        write_and_pack(2, idx, file)

        for i, name in enumerate(data_col):
            d = dataframe[name].values
            write_and_pack(start_of_data + i, d, file)

        for i, name in enumerate(label_cols):
            d = dataframe[name].values
            write_and_pack(start_of_labels + i, d, file)

    return fname

if __name__ == "__main__":
    import pandas as pd
    import string

    np.random.seed(12345)

    test_data = pd.DataFrame(np.random.rand(100, 4), columns=["x0", "y0", "z0", "col1"])

    # label columns still don't work
    # s = [s for s in string.ascii_letters + string.digits]
    # ss = ["".join(np.random.choice(s, size=np.random.randint(5, 10))) for i in range(100)]
    # test_data["label"] = ss

    export_mesh("test.am", test_data)  # , label_cols=["label"])
