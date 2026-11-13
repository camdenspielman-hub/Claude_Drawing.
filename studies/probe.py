"""Ask the light logic what it would mix at a few spots on an object (a check on the logic, not a picture)."""
import sys
import numpy as np
from still_life import scene as L
from still_life.paint import cast, studio_for

sc = L.Scene(scale=0.25)
objs = cast(sc)
sk = objs["skull"]
core = next(f for f in sk.forms if f.name == "skull_core")
lt = sk.lt
for x in (2.5, 4.5, 8.6):
    print("x =", x)
    for th in np.radians([90, 60, 30, 0, -30, -60]):       # from crown round the near flank to underneath
        P, N = core.surf(np.array([x]), np.array([np.pi / 2 - th]))
        c = core.colour(P, N, np.array([x]), np.array([np.pi / 2 - th]))
        ndl = float(N[0] @ lt.light.L)
        vis = float(lt.light.lit(P, ("skull:core", "under:"))[0])
        amb = lt.ambient(N, P)[0]
        bn = lt.bounce(P, N, ("skull:core", "under:"))[0]
        op = float(np.atleast_1d(lt.light.open_to_room(P, N, ("skull:core", "under:"), reach=0.8))[0])
        print("  N=(%5.2f %5.2f %5.2f) ndl %5.2f lit %4.2f open %4.2f amb %s bounce %s -> %s" % (
            *N[0], ndl, vis, op, np.round(amb, 3), np.round(bn, 3), np.round(c[0] * 255).astype(int)))
