#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __init__.py
"""
Convert colormaps to amira format
Copyright (c) 2018, David Hoffman
"""

import os
import matplotlib.cm
import colorcet


header = """# AmiraMesh 3D ASCII 2.0


define Lattice {N:d}

Parameters {{
    ContentType "Colormap"
    MinMax 0 {N:d}{extra:}
}}

Lattice {{ float[4] Data }} @1

# Data section follows
@1
"""

qualitative_str = """
    Interpolate 0
    OutOfBoundsBehavior "CycleRight"
    LabelField 1
"""

# for Label Field
# Parameters {
#     MinMax 1 8,
#     ContentType "Colormap",
#     Interpolate 0
#     OutOfBoundsBehavior "CycleRight"
#     LabelField 1
# }


def cm_to_amira(cmap, filename, extra=""):
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)
    filename = os.path.splitext(filename)[0] + ".am"
    with open(filename, "w") as fp:
        fp.write(header.format(N=cmap.N, extra=extra))
        # convert numbers to strings
        str_cmap = ([str(c) for c in cmap(i / (cmap.N - 1))] for i in range(cmap.N))
        str_cmap = [" ".join(color) for color in str_cmap]
        fp.write("\n".join(str_cmap))


if __name__ == "__main__":
    qualitative = {
        "Pastel1",
        "Pastel2",
        "Paired",
        "Accent",
        "Dark2",
        "Set1",
        "Set2",
        "Set3",
        "tab10",
        "tab20",
        "tab20b",
        "tab20c",
    }

    for cmap_name, cmap in matplotlib.cm.cmap_d.items():
        if cmap_name in qualitative:
            extra = qualitative_str
            print(cmap_name, "qualitative")
        else:
            extra = ""
            print(cmap_name)
        cm_to_amira(cmap, "colormaps/" + cmap_name, extra=extra)

    for cmap_name, cmap in colorcet.cm.items():
        cm_to_amira(cmap, "colormaps/" + cmap_name, extra="")
        print(cmap_name)
