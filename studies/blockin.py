"""Compositional block-in: every object as a simple stand-in solid, laid in broadly."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from atelier.canvas import Paint, Brush
from atelier.geom import Ball, Capsule, Disc, Quad, normalize, smoothstep, frame
from atelier.shading import lin
from atelier.brushwork import paint_surface, fill_field
from atelier import forms
from atelier.stencil import Stencil
from still_life import scene as L

sc = L.Scene(scale=0.4)
cv, cam, lt, light, rng, px = sc.cv, sc.cam, sc.lt, sc.light, sc.rng, sc.px
W, H = sc.W, sc.H
t0 = time.time()
up = np.array([0.0, 1.0, 0.0])

# ---------------- stand-ins
things = []   # (name, surf, urange, vrange, albedo, spec)
def add(name, surf, ur, vr, col, spec=0.0, gloss=40):
    things.append(dict(name=name, surf=surf, ur=ur, vr=vr, col=lin(col), spec=spec, gloss=gloss))
TWO_PI = 2 * np.pi
HG = L.HOURGLASS
wood = "#5a3418"
add("hg_base_side", forms.revolution(HG, up, lambda v: np.full_like(v, 4.4)), (0, TWO_PI), (0, 1.0), wood, 0.1)
add("hg_base_top", forms.disc(HG + [0, 1.0, 0], up, 4.4), (0, TWO_PI), (0, 1), wood, 0.1)
add("hg_top_side", forms.revolution(HG, up, lambda v: np.full_like(v, 4.4)), (0, TWO_PI), (15.0, 16.0), wood, 0.1)
add("hg_top_top", forms.disc(HG + [0, 16.0, 0], up, 4.4), (0, TWO_PI), (0, 1), wood, 0.1)
glass_prof = forms.Profile([(1.0, 0.0), (1.05, 1.6), (1.6, 2.5), (3.2, 2.95), (4.6, 2.8), (6.4, 1.5), (7.5, 0.45), (8.0, 0.28),
                            (8.5, 0.45), (9.6, 1.5), (11.4, 2.8), (12.8, 2.95), (14.4, 2.5), (14.95, 1.6), (15.0, 0.0)])
add("hg_glass", forms.revolution(HG, up, glass_prof), (0, TWO_PI), (1.0, 15.0), "#59544a", 0.5, 120)
posts = []
for phi in np.radians([20, 140, 260]):
    b = HG + [3.6 * np.cos(phi), 1.0, 3.6 * np.sin(phi)]
    posts.append(b)
    add("hg_post", forms.revolution(b, up, lambda v: 0.42 + 0.12 * np.sin(v * 2.2)), (0, TWO_PI), (0, 14.0), wood, 0.15)
yaw = L.SKULL_YAW
fwd = np.array([-np.cos(yaw), 0, np.sin(yaw)])
side = np.cross(up, fwd)
rot = np.stack([fwd, up, side], axis=1)
SK = L.SKULL + [0, 3.0, 0]
add("skull", forms.ellipsoid(SK, (5.0, 2.9, 3.4), rot), (0, TWO_PI), (-np.pi / 2, np.pi / 2), "#d8cba8", 0.05)
add("yarn", forms.sphere(L.YARN, L.YARN_R), (0, TWO_PI), (-np.pi / 2, np.pi / 2), "#3d5f73")
crng = np.random.default_rng(11)
coins = []
metals = ["#c9a03c"] * 9 + ["#b9b4a8"] * 7 + ["#9a5a36"] * 6
crng.shuffle(metals)
for i, col in enumerate(metals):
    a = crng.uniform(0, 2 * np.pi); d = 4.6 * np.sqrt(crng.uniform(0, 1))
    hgt = 0.12 + 1.1 * max(0, 1 - d / 3.6) * crng.uniform(0.3, 1)
    tilt = normalize([crng.normal(0, 0.22), 1, crng.normal(0, 0.22)])
    coins.append((L.COINS + [1.35 * d * np.cos(a), hgt, 0.8 * d * np.sin(a)], tilt, crng.choice([1.0, 1.15, 1.3, 1.5]), col))
coins.sort(key=lambda c: c[0][1])
for c, n, r, col in coins:
    add("coin", forms.disc(c, n, r), (0, TWO_PI), (0, 1), col, 0.6, 30)
OR, n_cut = L.ORANGE, normalize([-0.12, 0.90, 0.42])
e1, e2 = frame(n_cut)
rot_o = np.stack([e1, -n_cut, e2], axis=1)     # local +y points into the dome
add("orange_peel", forms.ellipsoid(OR, (L.ORANGE_R,) * 3, rot_o), (0, TWO_PI), (0, np.pi / 2), "#d9741a", 0.15, 25)
add("orange_cut", forms.disc(OR, n_cut, L.ORANGE_R), (0, TWO_PI), (0, 1), "#f0a22e", 0.2, 40)
ax = L.BLADE_AXIS
bw = normalize(np.cross(ax, [0.45, 0, 0.9]))       # blade width direction
bn = np.cross(bw, ax)
add("blade", forms.plane(L.BLADE - 1.1 * bw, 2.2 * bw, 11.0 * ax), (0, 1), (0, 1), "#8d9094", 0.9, 60)
add("guard", forms.revolution(L.BLADE + 11.0 * ax - 2.6 * bw, bw, lambda v: np.full_like(v, 0.35)), (0, TWO_PI), (0, 5.2), "#a8842e", 0.6)
add("grip", forms.revolution(L.BLADE + 11.35 * ax, ax, lambda v: 0.8 + 0.22 * np.sin(v / 7.2 * np.pi)), (0, TWO_PI), (0, 7.2), "#2e1c10", 0.2)
add("pommel", forms.sphere(L.BLADE + 19.3 * ax, 0.95), (0, TWO_PI), (-np.pi / 2, np.pi / 2), "#a8842e", 0.6)

nA0, nA1 = L.YARN + [-5.5, 6.0, -3.0], L.YARN + [7.5, -4.0, 5.5]
nB0, nB1 = L.YARN + [5.0, 6.5, -3.5], L.YARN + [-6.0, -3.9, 4.5]
for a_, b_ in ((nA0, nA1), (nB0, nB1)):
    d_ = b_ - a_
    add("needle", forms.revolution(a_, d_, lambda v: np.full_like(v, 0.2)), (0, TWO_PI), (0, np.linalg.norm(d_)), "#d9c9a0", 0.3)
from atelier.geom import catmull
thread = catmull(np.array([[-12.5, 0.15, 5.6], [-10.0, 0.15, 9.5], [-7.5, 0.15, 13.0], [-3.0, 0.15, 14.6], [1.5, 0.15, 13.2],
                           [5.0, 0.15, 12.2], [8.5, 0.15, 14.6], [12.5, 0.15, 15.6], [16.2, 0.15, 13.0], [17.4, 0.15, 7.5], [16.9, 0.15, 2.6]]), 0.4)

# ---------------- what blocks the light
light.add(Disc("hg", HG + [0, 16.0, 0], up, 4.4), Disc("hg", HG + [0, 1.0, 0], up, 4.4),
          Ball("hg", HG + [0, 4.0, 0], 2.9, opacity=0.5), Ball("hg", HG + [0, 12.0, 0], 2.9, opacity=0.35),
          Ball("skull", SK, 1.0, scale=(5.0, 2.9, 3.4), rot=rot), Ball("yarn", L.YARN, L.YARN_R),
          Ball("orange", OR, L.ORANGE_R),
          Quad("blade", [L.BLADE - 1.1 * bw, L.BLADE + 1.1 * bw, L.BLADE + 1.1 * bw + 11 * ax, L.BLADE - 1.1 * bw + 11 * ax]),
          Capsule("blade", L.BLADE + 11.35 * ax, L.BLADE + 19.3 * ax, 0.9),
          Capsule("blade", L.BLADE + 11.0 * ax - 2.6 * bw, L.BLADE + 11.0 * ax + 2.6 * bw, 0.35))
for b in posts:
    light.add(Capsule("hg", b, b + [0, 14.0, 0], 0.45))
for c, n, r, col in coins:
    light.add(Disc("coin", c, n, r))

# ---------------- background
def bg_col(xy):
    x, y = xy[:, 0] / W, xy[:, 1] / H
    glow = np.exp(-(((x - 0.72) / 0.42) ** 2 + ((y - 0.30) / 0.5) ** 2))
    v = 0.035 + 0.11 * glow
    return np.stack([v * 1.08, v * 0.98, v * 0.78], 1)
big = Brush(kind="bristle", radius=px(46), nbristle=110, bristle_r=px(3.4), jitter=0.1, thick=1.3, pickup=0.5, aspect=0.45, angle=np.pi / 2)
for _ in range(2):
    fill_field(cv, (0, 0, W, H * 0.66), bg_col, lambda xy: 0.5 + 0.4 * np.sin(xy[:, 0] / px(300)), big, rng,
               length=px(260), spacing=px(26), curl=0.12)
cv.dry_now()

# ---------------- table
table = forms.plane([0, 0, 0], [1.0, 0, 0], [0, 0, 1.0], flip=True)
oak = lin("#5f4830")
def table_col(P, N, u, v):
    g = 0.75 + 0.5 * forms.fbm3(P * [0.35, 1, 5.0], 1.0, 3, seed=3)
    fall = np.exp(-((P[:, 0] + 2) / 30) ** 2) * smoothstep(-24, 2, P[:, 2]) * 0.9 + 0.10
    return lt.shade(oak * g[:, None] * fall[:, None], P, N)
tb = Brush(kind="bristle", radius=px(22), nbristle=60, bristle_r=px(3), jitter=0.15, thick=1.3, pickup=0.5, aspect=0.5, angle=np.pi / 2)
paint_surface(cv, cam, table, table_col, tb, rng, urange=(-45, 45), vrange=(L.TABLE_BACK, L.TABLE_FRONT), length=px(170), spacing=px(15), grid=60)
cv.dry_now()
paint_surface(cv, cam, table, table_col, Brush(radius=px(9), hardness=0.3, thick=1.0, pickup=0.5), rng,
              urange=(-30, 34), vrange=(-24, L.TABLE_FRONT), length=px(50), spacing=px(8), grid=80, detail=(0.035, px(16)))
front = forms.plane([0, 0, L.TABLE_FRONT], [1.0, 0, 0], [0, -1.0, 0], flip=True)
def front_col(P, N, u, v):
    g = 0.75 + 0.5 * forms.fbm3(P * [0.35, 3.0, 1], 1.0, 3, seed=5)
    return lt.shade(oak * 0.8 * g[:, None], P, N)
paint_surface(cv, cam, front, front_col, tb, rng, urange=(-45, 45), vrange=(0, L.TABLE_DROP), length=px(170), spacing=px(15), grid=40)
cv.dry_now()

# ---------------- things, far to near
order = sorted(things, key=lambda t: -cam.depth(t["surf"](np.array([np.mean(t["ur"])]), np.array([np.mean(t["vr"])]))[0][0]))
for t in order:
    st = Stencil.from_surface(cam, t["surf"], t["ur"], t["vr"], W, H, grid=64)
    def col(P, N, u, v, t=t):
        ao = 0.4 + 0.6 * smoothstep(0.0, 2.0, P[:, 1])
        return lt.shade(t["col"], P, N, skip=(t["name"].split("_")[0],), spec=t["spec"], gloss=t["gloss"], ao=ao)
    small = t["name"] in ("coin", "guard", "hg_post", "pommel", "blade", "needle")
    r1 = px(7 if small else 16)
    paint_surface(cv, cam, t["surf"], col, Brush(radius=r1, hardness=0.45, thick=1.4, pickup=0.5), rng,
                  urange=t["ur"], vrange=t["vr"], length=r1 * 4, spacing=r1 * 0.7, stencil=st, grid=40)
    paint_surface(cv, cam, t["surf"], col, Brush(radius=px(3.5), hardness=0.5, thick=1.2, pickup=0.6), rng,
                  urange=t["ur"], vrange=t["vr"], length=px(16), spacing=px(3.2), stencil=st, grid=40)
    cv.dry_now()

from atelier.brushwork import line
line(cv, cam.project(thread), Paint.opaque("#4f7f93"), Brush(radius=px(8), hardness=0.6, thick=1.5), lo=0.9)
print("strokes", cv.nstrokes, "time %.1fs" % (time.time() - t0))
out = os.path.join(os.path.dirname(__file__), "out")
cv.save(os.path.join(out, "blockin.png"), relief=0.15)
