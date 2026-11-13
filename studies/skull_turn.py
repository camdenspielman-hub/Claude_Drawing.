"""Turnaround studies of the sculpted skull: the same clay model, laid in broadly with the brush
from the side, from above and from the front, to check the anatomy before it goes in the picture.

    python3 -m studies.skull_turn
"""
import os
import sys

import numpy as np
from PIL import Image

from atelier.canvas import Canvas
from atelier.geom import Camera
from atelier.studio import Studio
from still_life import scene as L
from still_life.skull import Skull

VIEWS = {"side": ([0.0, 0.1, 1.0], 62.0), "above": ([0.0, 1.0, 0.12], 62.0), "front": ([1.0, 0.25, 0.55], 62.0),
         "back": ([-0.8, 0.3, 0.6], 62.0)}


def study(view, size=(560, 380), stages=("colour", "model")):
    sc = L.Scene(scale=1.0, crop=(0, 0, size[0], size[1]), ground="#4a3b2a")
    probe = Skull(sc)
    c = probe.centre
    d, dist = VIEWS[view]
    d = probe.M0 @ np.asarray(d, dtype=np.float64)          # given in the skull's own frame (x nose, y up, z its left)
    eye = c + dist * d / np.linalg.norm(d)
    sc.cam = Camera(eye=eye, target=c, focal=2700.0, W=sc.W, H=sc.H)
    sc.lt.cam = sc.cam
    sc.s = 0.5                                              # brush sizes as for a half-size picture
    sk = Skull(sc)
    sk.lt = sc.lt
    sc.objs = {"skull": sk}
    st = Studio(sc)
    st.add(*sk.forms)
    st.prepare()
    st.run(stages=stages)
    return sc.cv.array()


if __name__ == "__main__":
    views = sys.argv[1:] or list(VIEWS)
    tiles = [Image.fromarray((np.clip(study(v), 0, 1) * 255).astype(np.uint8)) for v in views]
    W = sum(t.width for t in tiles[:2])
    rows = [tiles[i:i + 2] for i in range(0, len(tiles), 2)]
    sheet = Image.new("RGB", (W, sum(r[0].height for r in rows)))
    y = 0
    for r in rows:
        x = 0
        for t in r:
            sheet.paste(t, (x, y))
            x += t.width
        y += r[0].height
    os.makedirs("studies/out", exist_ok=True)
    sheet.save("studies/out/skull_turn.png")
    print("saved studies/out/skull_turn.png", sheet.size)
