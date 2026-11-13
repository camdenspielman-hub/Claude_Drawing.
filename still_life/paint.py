"""Paint the still life.

    python3 -m still_life.paint                     the whole picture
    python3 -m still_life.paint --only skull        the setting and one object, in a window round it
    python3 -m still_life.paint --scale 0.4         a quick small version
"""
import argparse
import os
import time

import numpy as np

from atelier.canvas import Recorder
from atelier.stencil import Stencil
from . import scene as L
from . import setting
from .coins import Coins
from .dagger import Dagger
from .hourglass import Hourglass
from .orange import Orange
from .skull import Skull
from .yarn import Yarn, Thread

MAKERS = {"skull": Skull, "hourglass": Hourglass, "yarn": Yarn, "thread": Thread, "coins": Coins,
          "orange": Orange, "dagger": Dagger}
ORDER = ["hourglass", "dagger", "yarn", "skull", "coins", "orange", "thread"]   # far to near


def build(sc, names):
    objs = {n: MAKERS[n](sc) for n in ORDER if n in MAKERS and n in names}
    for o in objs.values():
        sc.light.add(*o.occluders())
    sc.lt.pool = setting.window_pool
    return objs


def window_for(names, pad=90):
    """The part of the full canvas a set of objects (and their near surroundings) occupies."""
    probe = L.Scene(scale=0.25)
    objs = build(probe, names)
    box = None
    for o in objs.values():
        b = np.array(o.prepare().bbox()) / 0.25
        box = b if box is None else np.array([min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])])
    x0, y0, x1, y1 = box
    return (int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(L.W0, x1 + pad * 2)), int(min(L.H0, y1 + pad)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default="out/still_life.png")
    ap.add_argument("--frames", default=None, help="folder to save time-lapse frames in")
    ap.add_argument("--zoom", type=float, default=None, help="resize the saved image")
    args = ap.parse_args()

    names = [n for n in ORDER if n in MAKERS] if args.only is None else args.only
    crop = window_for(names) if args.only is not None else None
    sc = L.Scene(scale=args.scale, crop=crop)
    if args.frames:
        sc.cv.recorder = Recorder(args.frames, area_per_frame=sc.W * sc.H * 0.05, stride=3)
    t0 = time.time()
    objs = build(sc, names)
    sc.objs = objs                      # so one object can ask where another stands
    reserve = Stencil.empty(sc.W, sc.H)
    for o in objs.values():
        o.prepare()
        reserve = reserve | getattr(o, "reserve", o.stencil)
    setting.paint(sc, reserve=reserve, objects=list(objs.values()))
    print("setting done: %d strokes, %.0fs" % (sc.cv.nstrokes, time.time() - t0))
    for n in ORDER:
        if n in objs:
            objs[n].paint()
            print("%s done: %d strokes, %.0fs" % (n, sc.cv.nstrokes, time.time() - t0))
    if sc.cv.recorder is not None:
        sc.cv.recorder.snap(sc.cv)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sc.cv.save(args.out, relief=0.12, tooth_amt=0.05, scale=args.zoom)
    print("saved", args.out, (sc.W, sc.H))


if __name__ == "__main__":
    main()
