"""Paint the still life.

    python3 -m still_life.paint                     the whole picture
    python3 -m still_life.paint --only skull        the same picture, but only a window round one object
    python3 -m still_life.paint --scale 0.4         a quick small version
"""
import argparse
import importlib
import os
import time

import numpy as np

from atelier.canvas import Recorder
from atelier.studio import Studio, STAGES
from . import scene as L
from .setting import Setting

# name -> (module, class). Objects whose module is not written yet are simply left out.
CAST = {"hourglass": ("hourglass", "Hourglass"), "dagger": ("dagger", "Dagger"), "yarn": ("yarn", "Yarn"),
        "skull": ("skull", "Skull"), "coins": ("coins", "Coins"), "orange": ("orange", "Orange"),
        "strand": ("yarn", "Strand")}


def cast(sc, names=None):
    objs = {"setting": Setting(sc)}
    for name, (mod, cls) in CAST.items():
        if names is not None and name not in names:
            continue
        try:
            m = importlib.import_module("still_life." + mod)
        except ImportError:
            continue
        if hasattr(m, cls) and getattr(m, "STUDIO_READY", True):
            objs[name] = getattr(m, cls)(sc)
    sc.objs = objs
    for o in objs.values():
        sc.light.add(*o.occluders())
        sc.lt.glows.extend(o.glows())
    for o in objs.values():           # each object works with only the neighbours that matter to it
        if hasattr(o, "centre"):
            o.lt = sc.lt.near(o.centre, o.reach)
    return objs


def studio_for(sc, objs, focus_on="skull"):
    focus = None
    if focus_on in objs:
        focus = sc.cam.project(objs[focus_on].world([8.0, 1.2, 1.2]))
    st = Studio(sc, focus=focus)
    groups = []
    for o in objs.values():
        groups.extend(o.groups())
        o.register(st)
    for _, fs in sorted(groups, key=lambda g: -g[0]):       # far things first
        st.add(*fs)
    return st


def window_for(names, pad=110):
    """The part of the full canvas a set of objects occupies, with some surroundings."""
    probe = L.Scene(scale=0.2)
    objs = cast(probe, names)
    st = studio_for(probe, objs, focus_on=None)
    st.prepare()
    box = None
    for n in names:
        for f in objs[n].forms:
            if not f.region.any():
                continue
            b = np.array(f.stencil.bbox()) / 0.2
            box = b if box is None else np.array([min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])])
    x0, y0, x1, y1 = box
    return (int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(L.W0, x1 + pad * 1.6)), int(min(L.H0, y1 + pad)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="paint just a window round these objects")
    ap.add_argument("--alone", action="store_true", help="with --only: leave the other objects out entirely")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default="out/still_life.png")
    ap.add_argument("--frames", default=None, help="folder to save time-lapse frames in")
    ap.add_argument("--stages", nargs="*", default=list(STAGES))
    ap.add_argument("--steps", default=None, help="also save the picture after each stage, with this file prefix")
    args = ap.parse_args()

    crop = window_for(args.only) if args.only else None
    sc = L.Scene(scale=args.scale, crop=crop)
    if args.frames:
        sc.cv.recorder = Recorder(args.frames, area_per_frame=sc.W * sc.H * 0.05, stride=3)
    t0 = time.time()
    objs = cast(sc, args.only if (args.only and args.alone) else None)
    st = studio_for(sc, objs)
    st.prepare()
    print("%d forms outlined, %.0fs" % (len(st.forms), time.time() - t0), flush=True)

    def report(stage):
        print("%-7s %8d strokes  %.0fs" % (stage, sc.cv.nstrokes, time.time() - t0), flush=True)
        if args.steps:
            sc.cv.save("%s_%s.jpg" % (args.steps, stage), relief=0.10)
    st.run(stages=args.stages, report=report)
    if sc.cv.recorder is not None:
        sc.cv.recorder.snap(sc.cv)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sc.cv.save(args.out, relief=0.12, tooth_amt=0.05)
    print("saved", args.out, (sc.W, sc.H), flush=True)


if __name__ == "__main__":
    main()
