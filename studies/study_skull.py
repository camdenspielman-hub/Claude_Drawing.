"""Study: the skull's construction, each part in a flat tint, to check the drawing."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from atelier.canvas import Paint, Brush
from atelier.brushwork import paint_surface
from atelier.stencil import Stencil
from still_life import scene as L
from still_life.skull import Skull, nasal_opening

yaw = float(sys.argv[1]) if len(sys.argv) > 1 else None
probe = L.Scene(scale=0.1)
sk0 = Skull(probe, yaw=None if yaw is None else np.radians(yaw))
# find the skull on the full canvas and paint only that window
full = L.Scene.__new__(L.Scene)
from atelier.geom import Camera
cam_full = Camera(eye=probe.cam.eye, target=probe.cam.eye + probe.cam.f, focal=5100.0, W=L.W0, H=L.H0, cx=L.W0 / 2, cy=L.H0 / 2 + 100)
pts = []
for p in sk0.parts:
    U, V = np.meshgrid(np.linspace(*p.ur, 12), np.linspace(*p.vr, 12), indexing="ij")
    pts.append(cam_full.project(p.surf(U, V)[0]).reshape(-1, 2))
pts = np.concatenate(pts)
x0, y0 = np.floor(pts.min(0) - 60).astype(int); x1, y1 = np.ceil(pts.max(0) + 60).astype(int)
print("skull window on the canvas:", (x0, y0, x1, y1), "size", x1 - x0, y1 - y0)
sc = L.Scene(crop=(x0, y0, x1, y1), ground="#3a3026")
sk = Skull(sc, yaw=None if yaw is None else np.radians(yaw))
cv, cam, rng = sc.cv, sc.cam, sc.rng
sc.light.add(*sk.occluders())
t0 = time.time()
# a table and a dark behind, so the bone has something to sit against
from atelier import forms
from atelier.shading import lin
from atelier.brushwork import fill_field
for _ in range(2):
    fill_field(cv, (0, 0, sc.W, sc.H), lambda xy: np.tile([0.085, 0.072, 0.055], (len(xy), 1)), lambda xy: 0.3 + 0 * xy[:, 0],
               Brush(radius=14, hardness=0.4, thick=1.3), rng, length=70, spacing=8)
cv.dry_now()
table = forms.plane([0, 0, 0], [1.0, 0, 0], [0, 0, 1.0], flip=True)
oak = lin("#5f4830")
def table_col(P, N, u, v):
    g = 0.8 + 0.4 * forms.fbm3(P * [0.35, 1, 5.0], 1.0, 3, seed=3)
    return sc.lt.shade(oak * g[:, None], P, N)
for r, det in ((9, None), (9, None), (3.5, (0.02, 10)), (1.6, (0.02, 5))):
    paint_surface(cv, cam, table, table_col, Brush(radius=r, hardness=0.4, thick=1.3, pickup=0.5), rng, urange=(-6, 18), vrange=(-9, 12),
                  length=r * 6, spacing=r * 0.8, grid=90, detail=det)
cv.dry_now()
sk.paint()
print("strokes", cv.nstrokes, "time %.1fs" % (time.time() - t0))
out = os.path.join(os.path.dirname(__file__), "out")
cv.save(os.path.join(out, "study_skull%s.png" % ("" if yaw is None else "_%d" % yaw)), scale=1.5)
