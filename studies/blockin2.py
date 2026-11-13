"""Second composition, blocked in with stand-in solids through the new layered way of working."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from atelier import forms
from atelier.geom import Ball, Capsule, Disc, Quad, Glow, normalize, smoothstep, frame
from atelier.shading import lin
from atelier.studio import Studio, Form
from still_life import scene as L

sc = L.Scene(scale=float(sys.argv[1]) if len(sys.argv) > 1 else 0.4)
cam, lt, light = sc.cam, sc.lt, sc.light
up = np.array([0.0, 1.0, 0.0]); TWO_PI = 2 * np.pi
st = Studio(sc, focus=cam.project(L.SKULL + [-2.5, 3.0, 2.0]))

def pool(P):
    side = np.exp(-((P[:, 0] - 1.0) / 26.0) ** 2)
    depth = smoothstep(-26.0, 6.0, P[:, 2])
    return 0.10 + 0.90 * side * (0.30 + 0.70 * depth)
lt.pool = pool
OAK, WALL = lin("#5c4630"), lin("#6f6850")
lt.table_look = lambda Q: OAK * (lt.key * 0.66 * pool(Q)[:, None] + 0.10)
lt.wall_look = lambda Q: WALL * (lt.key * 0.56 * pool(Q)[:, None] + 0.08)

def matte(albedo, skip, **kw):
    return lambda P, N, u, v: lt.shade(albedo, P, N, skip=skip, **kw)

# wall and table
wall = forms.plane([0, 0, L.WALL_Z], [1.0, 0, 0], [0, 1.0, 0])
st.add(Form("wall", wall, (-60, 60), (0, 46), matte(WALL, ()), size=26, grid=40, finish=0.3, along="u", broken=0.06))
table = forms.plane([0, 0, 0], [1.0, 0, 0], [0, 0, 1.0], flip=True)
st.add(Form("table", table, (-46, 46), (L.TABLE_BACK, L.TABLE_FRONT), matte(OAK, ()), size=14, grid=70, finish=0.8, along="u", broken=0.07))
front = forms.plane([0, 0, L.TABLE_FRONT], [1.0, 0, 0], [0, -1.0, 0], flip=True)
st.add(Form("front", front, (-46, 46), (0, L.TABLE_DROP), matte(OAK * 0.6, ()), size=10, grid=40, finish=0.6, along="u"))

def thing(name, surf, ur, vr, col, size=6, **kw):
    st.add(Form(name, surf, ur, vr, matte(lin(col), (name.split("_")[0],), spec=kw.pop("spec", 0.0)), owner=name.split("_")[0], size=size, **kw))

# hourglass on its side
A, d = L.HOURGLASS, L.HOURGLASS_AXIS
glass = forms.Profile([(1.0, 1.3), (1.4, 2.4), (3.0, 3.1), (5.0, 3.0), (7.0, 1.6), (8.2, 0.5), (8.5, 0.32), (8.8, 0.5),
                       (10.0, 1.6), (12.0, 3.0), (14.0, 3.1), (15.6, 2.4), (16.0, 1.3)])
thing("hg_far", forms.revolution(A + 16 * d, d, lambda v: np.full_like(v, 4.6)), (0, TWO_PI), (0, 1.0), "#4a3220", 5)
thing("hg_glass", forms.revolution(A, d, glass), (0, TWO_PI), (1.0, 16.0), "#8f8a70", 6)
for k, ang in enumerate(np.radians([35, 125, 215, 305])):
    e1, e2 = frame(d)
    foot = A + d * 1.0 + 3.9 * (np.cos(ang) * e1 + np.sin(ang) * e2)
    thing("hg_post%d" % k, forms.revolution(foot, d, lambda v: np.full_like(v, 0.32)), (0, TWO_PI), (0, 15.0), "#4a3220", 2.2)
    light.add(Capsule("hg", foot, foot + 15 * d, 0.32))
thing("hg_near", forms.revolution(A, d, lambda v: np.full_like(v, 4.6)), (0, TWO_PI), (0, 1.0), "#4a3220", 5)
thing("hg_face", forms.disc(A, -d, 4.6), (0, TWO_PI), (0, 1), "#5a3e26", 6)
light.add(Ball("hg", A + 4.5 * d, 3.0, opacity=0.5), Ball("hg", A + 12.5 * d, 3.0, opacity=0.5),
          Capsule("hg", A, A + 1.0 * d, 4.5), Capsule("hg", A + 16 * d, A + 17 * d, 4.5))
lt.glows.append(Glow("hg", A + 0.3 * d, 4.0, lin("#5a3e26")))

# yarn
thing("yarn", forms.ellipsoid(L.YARN, (L.YARN_R, L.YARN_R * 0.94, L.YARN_R)), (0, TWO_PI), (-np.pi / 2, np.pi / 2), "#49738a", 8)
light.add(Ball("yarn", L.YARN, L.YARN_R)); lt.glows.append(Glow("yarn", L.YARN, L.YARN_R, lin("#49738a")))
for i, (a_, b_) in enumerate([(L.YARN + [-3.5, 9.5, -5.0], L.YARN + [2.5, -3.3, 6.5]), (L.YARN + [6.0, 5.0, -8.0], L.YARN + [-7.5, -3.55, 11.5])]):
    dn = b_ - a_
    thing("needle%d" % i, forms.revolution(a_, dn, lambda v: np.full_like(v, 0.11)), (0, TWO_PI), (0, np.linalg.norm(dn)), "#6c7076", 1.2, spec=0.5)
    light.add(Capsule("needle%d" % i, a_, b_, 0.11))

# skull stand-in
yaw = L.SKULL_YAW
fwd = np.array([-np.cos(yaw), 0, np.sin(yaw)]); side = np.cross(up, fwd)
rot = np.stack([fwd, up, side], axis=1)
SK = L.SKULL + [0, 3.0, 0]
thing("skull", forms.ellipsoid(SK, (5.2, 3.0, 3.5), rot), (0, TWO_PI), (-np.pi / 2, np.pi / 2), "#cdc3a8", 8)
light.add(Ball("skull", SK, 1.0, scale=(5.2, 3.0, 3.5), rot=rot)); lt.glows.append(Glow("skull", SK, 3.6, lin("#cdc3a8")))

# coins
crng = np.random.default_rng(4)
for i in range(22):
    a_ = crng.uniform(0, TWO_PI); r_ = 6.5 * np.sqrt(crng.uniform(0, 1))
    c = L.COINS + [1.25 * r_ * np.cos(a_), 0.06 + 0.25 * crng.uniform(0, 1) * (r_ < 3), 0.75 * r_ * np.sin(a_)]
    n = normalize([crng.normal(0, 0.12), 1, crng.normal(0, 0.12)]); R = crng.choice([0.95, 1.2, 1.5])
    thing("coin_%d" % i, forms.disc(c, n, R), (0, TWO_PI), (0, 1), crng.choice(["#b9963a", "#9a9890", "#7a5236"]), 2.6, spec=0.3)
    light.add(Disc("coin_%d" % i, c, n, R))

# orange
n_cut = normalize([-0.18, 0.80, 0.57]); e1, e2 = frame(n_cut)
thing("orange_peel", forms.ellipsoid(L.ORANGE, (L.ORANGE_R,) * 3, np.stack([e1, -n_cut, e2], axis=1)), (0, TWO_PI), (0, np.pi / 2), "#d86a12", 6)
thing("orange_cut", forms.disc(L.ORANGE, n_cut, L.ORANGE_R), (0, TWO_PI), (0, 1), "#f09a24", 6)
light.add(Ball("orange", L.ORANGE - 0.2 * L.ORANGE_R * n_cut, 0.95 * L.ORANGE_R)); lt.glows.append(Glow("orange", L.ORANGE, L.ORANGE_R, lin("#e8801a")))

# dagger stand-in: 21 cm of blade (2.5 buried), 10.5 cm of hilt
e, ax = L.BLADE, L.BLADE_AXIS
bw = normalize(np.cross(ax, normalize([-0.25, 0.15, 0.95]))); bn = np.cross(bw, ax)
thing("dagger_blade", forms.plane(e - 1.3 * bw, 2.6 * bw, 18.5 * ax), (0, 1), (0, 1), "#7d8086", 3, spec=0.6, min_facing=-2)
g = e + 18.5 * ax
thing("dagger_grip", forms.revolution(g + 0.5 * ax, ax, lambda v: np.full_like(v, 0.7)), (0, TWO_PI), (0, 8.0), "#2a1c14", 2.6)
thing("dagger_guard", forms.revolution(g - 3.4 * bw, bw, lambda v: np.full_like(v, 0.3)), (0, TWO_PI), (0, 6.8), "#3a3632", 2)
thing("dagger_pommel", forms.sphere(g + 9.4 * ax, 1.0), (0, TWO_PI), (-np.pi / 2, np.pi / 2), "#3a3632", 2.6)
light.add(Quad("dagger", [e - 1.3 * bw, e + 1.3 * bw, g + 1.3 * bw, g - 1.3 * bw]), Capsule("dagger", g, g + 9.4 * ax, 0.8),
          Capsule("dagger", g - 3.4 * bw, g + 3.4 * bw, 0.3))

st.softness_of.update({"wall": (0.5, 0.2), "table": (0.4, 0.0), "front": (0.3, 0), "yarn": (0.45, 0.0), "hg": (0.2, 0.2),
                       "skull": (0.1, 0.0), "orange": (0.15, 0.0), "dagger": (0.08, 0.1)})
t0 = time.time()
st.prepare()
print("prepared %d forms in %.1fs" % (len(st.forms), time.time() - t0))
out = os.path.join(os.path.dirname(__file__), "out"); os.makedirs(out, exist_ok=True)
def report(stage):
    print("%-7s %7d strokes  %.0fs" % (stage, sc.cv.nstrokes, time.time() - t0), flush=True)
    sc.cv.save(os.path.join(out, "b2_%s.jpg" % stage), relief=0.12)
st.run(report=report)
