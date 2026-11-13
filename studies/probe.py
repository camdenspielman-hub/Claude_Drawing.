"""Ask the skull what lies under a few spots of the canvas and what the light logic mixes there
(a check on the logic, not a picture). Spots are given in half-size picture pixels."""
import sys
import numpy as np
from still_life import scene as L
from still_life.paint import cast

sc = L.Scene(scale=0.5)
objs = cast(sc)
sk = objs["skull"]
pts = np.array([[float(a), float(b)] for a, b in zip(sys.argv[1::2], sys.argv[2::2])])
P, N = sk.seen(pts[:, 0], pts[:, 1])
g = sk.solid.group(P)
Pl = sk.local(P)
c = sk._colour(P, N, pts[:, 0], pts[:, 1])
lt = sk.lt
for i in range(len(pts)):
    print(pts[i], "group", g[i], "local", np.round(Pl[i], 2), "N.L %.2f" % float(N[i] @ lt.light.L),
          "sees %.2f" % sk.solid.sees(P[i:i+1], N[i:i+1], lt.light.L)[0],
          "open %.2f/%.2f" % (sk.solid.openness(P[i:i+1], N[i:i+1], 0.45 * sk.k)[0], sk.solid.openness(P[i:i+1], N[i:i+1], 2.6 * sk.k)[0]),
          "->", np.round(c[i] * 255).astype(int))
