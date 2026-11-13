"""Study: a ball on a table. Tuning how strokes model form, shadow and blending."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from atelier.canvas import Canvas, Paint, Brush
from atelier.geom import Camera, Light, Ball, normalize, smoothstep
from atelier.shading import Lighting, lin
from atelier.brushwork import paint_surface, fill_field

W, H = 1200, 900
rng = np.random.default_rng(3)
cv = Canvas(W, H, ground="#6b5238")
cam = Camera(eye=(0, 14, 40), target=(0, 3, 0), focal=2600, W=W, H=H)
light = Light(toward=(-0.55, 0.70, 0.45), soft=0.07)
C, R = np.array([0.0, 4.0, 0.0]), 4.0
light.add(Ball("ball", C, R))
lt = Lighting(light, cam)
t0 = time.time()

# background
def bg_col(xy):
    t = xy[:, 0] / W
    v = 0.10 + 0.10 * smoothstep(0.2, 1.0, t) + 0.02 * rng.normal(0, 1, len(xy))
    return np.stack([v * 1.05, v * 0.95, v * 0.80], 1)
fill_field(cv, (0, 0, W, H * 0.62), bg_col, lambda xy: 0.4 + 0.5 * np.sin(xy[:, 0] / 90 + xy[:, 1] / 70),
           Brush(kind="bristle", radius=22, nbristle=90, bristle_r=1.6, jitter=0.12, thick=1.2, pickup=0.5, aspect=0.4, angle=np.pi / 2),
           rng, length=110, spacing=14, curl=0.2)
cv.dry_now()

# table: a plane, strokes along the grain (x)
def table(u, v):
    P = np.stack([u, np.zeros_like(u), v], -1)
    return P, np.broadcast_to(np.array([0.0, 1.0, 0.0]), P.shape)
wood = lin("#7a5230")
def table_col(P, N, u, v):
    grain = 1 + 0.10 * np.sin(v * 9 + 2 * np.sin(u * 0.7)) + 0.06 * rng.normal(0, 1, len(P))
    return lt.shade(wood * grain[:, None], P, N)
paint_surface(cv, cam, table, table_col,
              Brush(kind="bristle", radius=9, nbristle=40, bristle_r=1.3, jitter=0.15, thick=1.3, pickup=0.5, aspect=0.5, angle=np.pi / 2),
              rng, urange=(-16, 16), vrange=(-22, 12), length=70, spacing=6, grid=40)
cv.dry_now()
# second, finer pass to firm up the shadow shape
paint_surface(cv, cam, table, table_col, Brush(radius=5, hardness=0.3, thick=0.9, pickup=0.5),
              rng, urange=(-8, 14), vrange=(-12, 6), length=30, spacing=4, grid=40)
# soften wet-in-wet
paint_surface(cv, cam, table, table_col, Brush(radius=7, hardness=0.2, thick=0.0, pickup=0.5, load=0.01),
              rng, urange=(-8, 14), vrange=(-12, 6), length=40, spacing=7, grid=40, paint=Paint.opaque("#000000"))
cv.dry_now()

# ball: longitude u, latitude v
def ball(u, v):
    n = np.stack([np.cos(v) * np.cos(u), np.sin(v), np.cos(v) * np.sin(u)], -1)
    return C + R * n, n
bone = lin("#d9cdb0")
def ball_col(P, N, u, v):
    # contact shadow: the table darkens what faces it closely
    ao = smoothstep(0.0, 2.5, P[:, 1])
    return lt.shade(bone, P, N, skip=("ball",), ao=0.35 + 0.65 * ao, spec=0.08, gloss=30)
# block-in, big brush
paint_surface(cv, cam, ball, ball_col, Brush(kind="bristle", radius=10, nbristle=45, bristle_r=1.3, jitter=0.08, thick=1.4, pickup=0.5, aspect=0.6, angle=np.pi / 2),
              rng, urange=(0, 2 * np.pi), vrange=(-np.pi / 2, np.pi / 2), length=46, spacing=6)
# modelling, smaller brush, across the first direction
paint_surface(cv, cam, ball, ball_col, Brush(radius=4.5, hardness=0.35, thick=1.0, pickup=0.55),
              rng, urange=(0, 2 * np.pi), vrange=(-np.pi / 2, np.pi / 2), along="v", length=26, spacing=3.5)
# blend
paint_surface(cv, cam, ball, ball_col, Brush(radius=6, hardness=0.2, thick=0.0, pickup=0.45, load=0.01),
              rng, urange=(0, 2 * np.pi), vrange=(-np.pi / 2, np.pi / 2), length=30, spacing=6, paint=Paint.opaque("#000000"))
cv.dry_now()
print("strokes", cv.nstrokes, "time %.2fs" % (time.time() - t0))
out = os.path.join(os.path.dirname(__file__), "out")
cv.save(os.path.join(out, "study_sphere.png"), relief=0.3, tooth_amt=0.1)
cv.save(os.path.join(out, "study_sphere_crop.png"), relief=0.3, tooth_amt=0.1, crop=(380, 330, 880, 730), scale=2)
