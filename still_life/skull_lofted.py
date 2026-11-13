"""The cat's skull, constructed from anatomy and checked against photographs of real ones.

Built in its own frame (cm): X runs from the back of the head (0) to the front teeth (10), Y is up
from the plane of the upper tooth row, Z is the animal's left. What makes it a cat and not a dog
or a fox: the face in front of the eye sockets is barely a fifth of the length; the sockets are
huge, face forward, and stay open behind; the cheekbone arches flare wider than the braincase;
the top is one arc; the lower jaw is two slim bars with a tall blade rising inside each arch.
"""
import numpy as np

from atelier import forms
from atelier.brushwork import Batch
from atelier.canvas import Brush, Paint
from atelier.forms import Table1D, Curve, fbm3, noise3
from atelier.geom import Ball, Capsule, Glow, normalize, smoothstep, catmull
from atelier.shading import lin
from atelier.studio import Form
from . import scene as L

TWO_PI = 2 * np.pi
MIRROR = np.diag([1.0, 1.0, -1.0])

# ---- the central body, as cross-sections from back to front
CORE = Table1D(
    [0.0, 0.25, 0.8, 1.6, 2.4, 3.2, 4.0, 4.6, 5.2, 5.9, 6.6, 7.3, 7.9, 8.5, 9.0, 9.4, 9.75, 10.0],
    w=[0.90, 1.75, 2.10, 2.28, 2.32, 2.25, 1.95, 1.62, 1.72, 2.05, 1.95, 1.72, 1.58, 1.46, 1.38, 1.30, 1.10, 0.72],
    yc=[1.60, 1.62, 1.72, 1.82, 1.88, 1.92, 1.98, 2.02, 2.04, 2.00, 1.90, 1.76, 1.58, 1.38, 1.22, 1.08, 0.86, 0.62],
    ht=[0.85, 1.45, 1.88, 2.10, 2.16, 2.16, 2.08, 1.98, 1.90, 1.82, 1.70, 1.52, 1.34, 1.14, 0.98, 0.86, 0.60, 0.32],
    hb=[0.90, 1.20, 1.42, 1.52, 1.55, 1.55, 1.52, 1.50, 1.50, 1.50, 1.48, 1.42, 1.30, 1.12, 0.97, 0.84, 0.62, 0.34],
    ez=[0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.82, 0.72, 0.62, 0.58, 0.58, 0.60, 0.70, 0.85],
    ey=[0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.88, 0.84, 0.80, 0.80, 0.80, 0.80, 0.85, 0.90],
    # underneath, the muzzle is squared off: a flat palate with upright sides down to the tooth row
    eb=[0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.70, 0.52, 0.45, 0.42, 0.42, 0.42, 0.42, 0.60, 0.85])

ORBIT_C = (6.55, 2.00)      # centre of the eye socket on the side of the head (X, Y)
ORBIT_R = (1.45, 1.32)
ORBIT_DEPTH = 0.60


def core_carve(x, y):
    """The hollows in the side of the head: the eye socket, and the shallower temple behind it."""
    r = np.sqrt(((x - ORBIT_C[0]) / ORBIT_R[0]) ** 2 + ((y - ORBIT_C[1]) / ORBIT_R[1]) ** 2)
    orbit = ORBIT_DEPTH * (1 - smoothstep(0.66, 1.0, r))
    rt = np.sqrt(((x - 4.5) / 1.5) ** 2 + ((y - 1.4) / 1.1) ** 2)
    temple = 0.20 * (1 - smoothstep(0.3, 1.0, rt))
    return np.maximum(orbit, temple)


def core_shape(x, y, z, yc):
    """The low crest along the midline of the crown, and the sharp one across the back of the head."""
    top = np.clip((y - yc) / 1.5, 0, 1)
    sagittal = 0.07 * np.exp(-(z / 0.16) ** 2) * smoothstep(3.6, 1.2, x) * top
    nuchal = 1 + 0.045 * np.exp(-((x - 0.32) / 0.13) ** 2)
    return yc + (y - yc) * nuchal + sagittal, z * nuchal


def nasal_opening(P_local):
    """True where the front of the muzzle is open: the bony nostril, heart-shaped, facing forward."""
    x, y, z = P_local[..., 0], P_local[..., 1], P_local[..., 2]
    rim = 1.96 - 2.2 * (x - 9.40)                         # dropping steeply from the nasal tips to the incisors
    half = 0.20 + 0.36 * np.clip((y - 0.72) / 1.0, 0, 1)  # wide above, narrow below
    return (x > 9.40) & (y > rim) & (np.abs(z) < half)


# ---- cheekbone arch (left): a stout rod, its lower edge level with the cheek teeth
_ARCH = np.array([
    # X     Y     Z     half-height
    [8.25, 0.80, 1.15, 0.18],     # grows out of the cheek
    [7.85, 0.80, 1.60, 0.46],     # deep under the front of the eye socket
    [7.30, 0.72, 2.20, 0.42],
    [6.70, 0.68, 2.75, 0.36],
    [6.10, 0.70, 3.15, 0.33],
    [5.50, 0.74, 3.40, 0.31],
    [4.80, 0.80, 3.50, 0.27],     # the widest part of the whole skull
    [4.10, 0.86, 3.42, 0.24],
    [3.50, 0.90, 3.15, 0.24],
    [3.05, 0.92, 2.82, 0.29],
    [2.72, 0.94, 2.46, 0.36],     # over the jaw hinge, broadening into a shelf
    [2.42, 0.98, 2.06, 0.44],
    [2.15, 1.04, 1.55, 0.46]])    # and running into the side of the braincase
ARCH = Curve(_ARCH[:, :3])
_seg = np.linalg.norm(np.diff(_ARCH[:, :3], axis=0), axis=1)
_ARCH_T = np.concatenate([[0], np.cumsum(_seg)]) / _seg.sum()

# ---- the ridge of bone along the upper tooth row, from the front teeth round to behind the carnassial
_RIDGE = np.array([
    # X      Z     Ytop  Ybot
    [10.08, 0.00, 0.46, 0.22],
    [10.06, 0.36, 0.46, 0.22],
    [10.00, 0.68, 0.48, 0.22],
    [9.86, 1.00, 0.56, 0.22],
    [9.52, 1.30, 0.86, 0.20],     # swelling over the root of the canine
    [9.20, 1.40, 0.90, 0.20],
    [8.75, 1.46, 0.80, 0.24],
    [8.30, 1.58, 0.76, 0.26],
    [7.80, 1.76, 0.74, 0.27],
    [7.30, 1.93, 0.72, 0.28],
    [6.85, 2.06, 0.66, 0.30],
    [6.60, 2.04, 0.56, 0.36]])
RIDGE_TOP = Curve(np.stack([_RIDGE[:, 0], _RIDGE[:, 2], _RIDGE[:, 1] - 0.30], 1), param=np.arange(len(_RIDGE)))
RIDGE_BOT = Curve(np.stack([_RIDGE[:, 0], _RIDGE[:, 3], _RIDGE[:, 1] + 0.02], 1), param=np.arange(len(_RIDGE)))

# ---- lower jaw (left half): a slim bar about 8 mm deep, with the blade that rises inside the arch
_JAW = np.array([
    # X     Ytop   Ybot   Z
    [9.92, -0.32, -0.92, 0.00],     # the chin, where the two halves join
    [9.84, -0.32, -1.00, 0.24],
    [9.45, -0.34, -1.16, 0.58],
    [8.80, -0.36, -1.20, 1.02],
    [8.00, -0.38, -1.20, 1.38],
    [7.20, -0.40, -1.20, 1.66],
    [6.50, -0.36, -1.22, 1.90],     # behind the last tooth
    [6.00, 0.10, -1.22, 2.05],
    [5.40, 0.80, -1.18, 2.18],
    [4.70, 1.40, -1.08, 2.28],
    [4.20, 1.62, -0.98, 2.33],      # tip of the blade (coronoid process)
    [3.75, 1.05, -0.92, 2.36],
    [3.35, 0.45, -0.88, 2.36],      # the notch
    [3.00, 0.42, -0.86, 2.34],      # the hinge
    [2.85, 0.10, -0.80, 2.30]])     # the little hook below it
JAW_TOP = Curve(_JAW[:, [0, 1, 3]], smooth=False)
JAW_BOT = Curve(_JAW[:, [0, 2, 3]])
JAW_FLOOR = -1.22


def _jaw_bulge(u):
    return np.interp(u, [0, 0.45, 0.6, 0.8, 1.0], [0.20, 0.22, 0.10, 0.04, 0.08])


def _spike(a, m, b, r0, power=0.75, flatten=None):
    """A canine: a stout tapering tooth from root a through m to tip b."""
    return forms.tube(Curve([a, m, b]), lambda v: r0 * np.clip(1 - v, 0, 1) ** power + 0.012, flatten=flatten)


def _peg(a, b, r0):
    """An incisor: a tiny peg with a rounded end."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return forms.tube(Curve([a, 0.5 * (a + b), b], smooth=False),
                      lambda v: r0 * np.sqrt(np.clip(1 - np.clip(v, 0, 1) ** 3, 0, 1)) + 0.008)


def _crown(xs, gum_y, tip_y, z0, z1, upper=True, thick=0.11):
    """A cheek tooth: one crown between its gum line and the jagged profile of its cusps."""
    xs = np.asarray(xs, dtype=np.float64)
    zs = z0 + (xs - xs[0]) / (xs[-1] - xs[0]) * (z1 - z0)
    gum = Curve(np.stack([xs, np.broadcast_to(gum_y, xs.shape), zs], 1), smooth=False, param=xs)
    tip = Curve(np.stack([xs, np.asarray(tip_y, dtype=np.float64), zs + 0.03], 1), smooth=False, param=xs)
    return forms.rails(gum, tip, [0, 0, 1], bulge=thick, edge=0.8) if upper else forms.rails(tip, gum, [0, 0, 1], bulge=thick, edge=0.8)


BONE = lin("#c6bda4")
BONE_STAIN = lin("#a88f5e")
BONE_THIN = lin("#b9b3a4")
ENAMEL = lin("#ddd6c2")


def _albedo(Pl, kind, gum=None):
    """Local colour of cleaned bone: matte warm ivory, greyer where it is thin, stained in the hollows."""
    if kind == "tooth":
        g = np.zeros(len(Pl)) if gum is None else gum
        tint = ENAMEL * (0.94 + 0.06 * fbm3(Pl * 9, 1, 2, 5))[:, None]
        tint = tint * (1 - 0.34 * g[:, None] * np.array([0.25, 0.55, 1.0]))
        return tint * (1 - 0.10 * (1 - g[:, None]) * np.array([1.0, 0.8, 0.5]))
    broad = smoothstep(0.45, 0.75, fbm3(Pl * 0.8, 1, 3, 11))[:, None]
    fine = fbm3(Pl * 6.5, 1, 3, 23)[:, None]
    pores = noise3(Pl * 30.0, 1.0, 31)[:, None]
    a = BONE * (1 - 0.42 * broad) + BONE_STAIN * 0.42 * broad
    if kind == "core":
        # the dome of the braincase is thin: a little greyer, cooler, as if light went into it
        thin = (smoothstep(0.2, 1.4, Pl[:, 1] - 2.2) * smoothstep(5.0, 3.6, Pl[:, 0]))[:, None]
        a = a * (1 - 0.35 * thin) + BONE_THIN * 0.35 * thin
        deep = (core_carve(Pl[:, 0], Pl[:, 1]) / ORBIT_DEPTH)[:, None]
        a = a * (1 - 0.30 * deep) + BONE_STAIN * 0.8 * 0.30 * deep
    if kind == "jaw":
        a = a * np.array([1.0, 0.965, 0.90])
    return a * (0.88 + 0.18 * fine) * (0.955 + 0.09 * pores)


class Skull:
    def __init__(self, sc, pos=None, yaw=None):
        self.sc = sc
        pos = L.SKULL if pos is None else pos
        yaw = L.SKULL_YAW if yaw is None else yaw
        fwd = np.array([-np.cos(yaw), 0.0, np.sin(yaw)])
        up = np.array([0.0, 1.0, 0.0])
        left = np.cross(up, fwd)
        self.k = L.SKULL_SCALE
        self.M0 = np.stack([fwd, up, left], axis=1)
        self.M = self.M0 * self.k
        self.t = np.asarray(pos, dtype=np.float64) - self.M @ np.array([5.0, JAW_FLOOR, 0.0])
        self.lt = None
        self.forms = []
        self._build()

    # ---- frames
    def world(self, P_local):
        return self.t + np.asarray(P_local, dtype=np.float64) @ self.M.T

    def local(self, P_world):
        return (np.asarray(P_world, dtype=np.float64) - self.t) @ self.M0 / self.k

    def place(self, surf, mirror=False):
        M = self.M @ MIRROR if mirror else self.M
        return forms.transformed(surf, M, self.t)

    # ---- what it blocks of the light, and what it throws back
    def occluders(self):
        k = self.k
        occ = [Ball("skull:core", self.world([2.5, 1.9, 0]), k, scale=(2.6, 2.1, 2.25), rot=self.M0),
               Ball("skull:core", self.world([5.6, 2.9, 0]), k, scale=(1.5, 0.95, 1.9), rot=self.M0),
               Ball("skull:core", self.world([6.5, 1.3, 0]), k, scale=(1.6, 1.0, 0.8), rot=self.M0),
               Ball("skull:core", self.world([8.7, 1.35, 0]), k, scale=(1.35, 1.1, 1.4), rot=self.M0)]
        occ.append(Ball("under:skull", self.world([5.6, 0.2, 0]), k, scale=(4.3, 1.1, 3.3), rot=self.M0))
        for sgn, s in ((1, "L"), (-1, "R")):
            m = np.array([1, 1, sgn])
            pts = ARCH(np.linspace(0, 1, 7)) * m
            for a, b in zip(pts[:-1], pts[1:]):
                occ.append(Capsule("skull:arch_" + s, self.world(a), self.world(b), 0.34 * k))
            jm = 0.5 * (JAW_TOP(np.linspace(0, 0.45, 5)) + JAW_BOT(np.linspace(0, 0.45, 5))) * m
            for a, b in zip(jm[:-1], jm[1:]):
                occ.append(Capsule("skull:jaw_" + s, self.world(a), self.world(b), 0.40 * k))
            occ.append(Ball("skull:jaw_" + s, self.world(np.array([4.6, 0.1, 2.2]) * m), k, scale=(1.5, 1.2, 0.25), rot=self.M0))
            occ.append(Capsule("skull:canine_up_" + s, self.world(np.array([9.33, 0.2, 1.12]) * m),
                               self.world(np.array([9.43, -0.75, 1.2]) * m), 0.16 * k))
        return occ

    def glows(self):
        return [Glow("skull", self.world([3.2, 2.0, 0]), 2.4 * self.k, BONE),
                Glow("skull", self.world([7.6, 1.6, 0]), 1.6 * self.k, BONE)]

    # ---- the forms, far side first
    def _build(self):
        sc = self.sc
        core = forms.loft(CORE, carve=core_carve, shape=core_shape)
        self.core_local = core
        arch = forms.tube(ARCH, lambda v: np.interp(v, _ARCH_T, _ARCH[:, 3]), flatten=([0, 0, 1], 0.5))
        # the two blunt tabs of bone that reach toward each other behind the eye and never quite meet
        jugal_spur = forms.tube(Curve([[5.78, 0.92, 3.34], [5.68, 1.28, 3.24], [5.58, 1.62, 3.10]], smooth=False),
                                lambda v: 0.40 * np.clip(1 - v, 0, 1) ** 1.1 + 0.03, flatten=([0, 0.25, 1], 0.22))
        brow_spur = forms.tube(Curve([[5.50, 3.16, 1.50], [5.46, 3.00, 1.80], [5.42, 2.78, 2.08]], smooth=False),
                               lambda v: 0.25 * np.clip(1 - v, 0, 1) ** 0.9 + 0.03, flatten=([0.1, 0.8, 0.6], 0.35))
        jaw = forms.rails(JAW_TOP, JAW_BOT, [0, 0, 1], bulge=_jaw_bulge, edge=0.9)
        self.jaw_local = jaw
        hinge = forms.tube(Curve([[2.98, 0.40, 1.80], [2.98, 0.42, 2.25], [2.98, 0.40, 2.70]], smooth=False),
                           lambda v: 0.17 + 0 * v)
        ridge = forms.rails(RIDGE_TOP, RIDGE_BOT, [0.35, 0, 1], bulge=0.025, edge=0.25)
        bulla = forms.ellipsoid([2.0, 0.05, 1.25], (0.80, 0.55, 0.50))
        up_canine = _spike([9.30, 0.62, 1.08], [9.48, -0.24, 1.19], [9.43, -0.86, 1.22], 0.25, power=0.62)
        lo_canine = _spike([9.58, -0.78, 0.72], [9.80, 0.00, 0.84], [9.76, 0.40, 0.90], 0.20, power=0.62)
        teeth_up = [_peg([x - 0.06, 0.52, z], [x - 0.01, tip, z], r)
                    for x, z, r, tip in [(10.06, 0.11, 0.085, 0.03), (10.04, 0.30, 0.09, 0.01), (9.99, 0.52, 0.12, -0.07)]]
        teeth_lo = [_peg([x - 0.08, -0.66, z], [x, -0.12, z], 0.07)
                    for x, z in [(9.98, 0.08), (9.96, 0.23), (9.92, 0.38)]]
        # upper cheek teeth: a minute peg, a pointed premolar, and the long shearing tooth under the eye socket
        cheek_up = [_crown([8.92, 8.86, 8.80], 0.50, [0.22, 0.12, 0.22], 1.34, 1.37, thick=0.06),
                    _crown([8.62, 8.52, 8.41, 8.33, 8.25, 8.16, 8.06], 0.55, [0.24, 0.12, -0.08, -0.16, -0.08, 0.10, 0.22],
                           1.44, 1.60, thick=0.17),
                    _crown([8.00, 7.91, 7.84, 7.76, 7.66, 7.54, 7.46, 7.38, 7.30, 7.20, 7.08, 6.96, 6.86], 0.58,
                           [0.22, 0.06, 0.00, 0.06, -0.10, -0.22, -0.25, -0.19, 0.00, -0.08, -0.11, 0.02, 0.18],
                           1.66, 1.98, thick=0.19)]
        cheek_lo = [_crown([8.92, 8.80, 8.70, 8.60, 8.48], -0.66, [-0.34, -0.26, -0.20, -0.26, -0.34], 1.00, 1.16, upper=False, thick=0.13),
                    _crown([8.40, 8.27, 8.17, 8.11, 8.05, 7.95, 7.85], -0.68, [-0.36, -0.28, -0.15, -0.11, -0.15, -0.28, -0.36],
                           1.16, 1.34, upper=False, thick=0.14),
                    _crown([7.78, 7.66, 7.58, 7.50, 7.36, 7.22, 7.14, 7.06, 6.86], -0.70,
                           [-0.36, -0.20, -0.10, -0.14, -0.26, -0.16, -0.10, -0.18, -0.34], 1.34, 1.58, upper=False, thick=0.14)]

        def add(name, surf, ur, vr, kind="bone", mirror=False, grid=40, size=4.5, both=False, along="u", **kw):
            placed = self.place(surf, bool(mirror))
            if both:
                placed = forms.toward_eye(placed, sc.cam)
            tag = "" if mirror is None else ("R" if mirror else "L")
            full = name + ("_" + tag if tag else "")
            f = Form("skull_" + full, placed, ur, vr, None, owner="skull", grid=grid, size=size, along=along, **kw)
            f.kind, f.inside, f.tag, f.base = kind, both, tag, name
            f.colour = self._colour(f)
            self.forms.append(f)
            return f

        def lower(m):
            add("bulla", bulla, (0, TWO_PI), (-np.pi / 2, np.pi / 2), mirror=m, grid=20, size=2.5, finish=0.5)
            for i, t in enumerate(cheek_lo):
                add("cheeklo%d" % i, t, (0, 1), (-1, 1), kind="tooth", mirror=m, grid=20, size=1.3, both=m, along="v", dead=False)
            for i, t in enumerate(teeth_lo):
                add("inclo%d" % i, t, (0, TWO_PI), (0, 1), kind="tooth", mirror=m, grid=12, size=1.1, along="v", dead=False)
            add("canine_lo", lo_canine, (0, TWO_PI), (0, 1), kind="tooth", mirror=m, grid=20, size=2.0, along="v", dead=False)
            add("hinge", hinge, (0, TWO_PI), (0, 1), kind="jaw", mirror=m, grid=14, size=1.6, finish=0.5)
            add("jaw", jaw, (0, 1), (-1, 1), kind="jaw", mirror=m, grid=60, size=3.6, both=m)

        def upper(m):
            for i, t in enumerate(cheek_up):
                add("cheekup%d" % i, t, (0, 1), (-1, 1), kind="tooth", mirror=m, grid=20, size=1.3, both=m, along="v", dead=False)
            for i, t in enumerate(teeth_up):
                add("incup%d" % i, t, (0, TWO_PI), (0, 1), kind="tooth", mirror=m, grid=12, size=1.1, along="v", dead=False)
            add("canine_up", up_canine, (0, TWO_PI), (0, 1), kind="tooth", mirror=m, grid=20, size=2.0, along="v", dead=False)

        lower(True)
        add("arch", arch, (0, TWO_PI), (0, 1), mirror=True, grid=50, size=3.0, along="v")
        upper(True)
        add("ridge", ridge, (0, 1), (-1, 1), mirror=True, grid=50, size=2.6, both=True)
        add("nose", core, (9.2, 10.0), (-1.4, 1.4), kind="cavity", mirror=None, grid=40, size=2.4, min_facing=-2.0,
            vis=lambda P, N, u, v: nasal_opening(self.local(P)), along="v", broken=0.0)
        add("core", core, (0.0, 10.0), (-np.pi, np.pi), kind="core", mirror=None, grid=90, size=5.0, along="v",
            vis=lambda P, N, u, v: ~nasal_opening(self.local(P)), outline_grid=160)
        add("brow", brow_spur, (0, TWO_PI), (0, 1), mirror=False, grid=16, size=2.0, along="v")
        lower(False)
        upper(False)
        add("ridge", ridge, (0, 1), (-1, 1), mirror=False, grid=50, size=2.6)
        add("arch", arch, (0, TWO_PI), (0, 1), mirror=False, grid=50, size=3.0, along="v")
        add("jugal", jugal_spur, (0, TWO_PI), (0, 1), mirror=False, grid=16, size=2.0, along="v")

    def _colour(self, f):
        """The colour rule for one part: bone or enamel under the room's light, darker in its hollows."""
        tag, base, kind = f.tag, f.base, f.kind
        skip = {"core": ("skull:core",), "brow": ("skull:core",), "arch": ("skull:arch_" + tag,),
                "jugal": ("skull:arch_" + tag,), "jaw": ("skull:jaw_" + tag,), "hinge": ("skull:jaw_" + tag,),
                "ridge": ("skull:core", "skull:canine_up_" + tag), "canine_up": ("skull:canine_up_" + tag,),
                "bulla": ()}.get(base, ())
        skip = skip + ("under:",)
        tooth = kind == "tooth"
        if base.startswith("cheekup"):
            gum_of = lambda u, v: smoothstep(-0.1, 0.5, v)
        elif base.startswith("cheeklo"):
            gum_of = lambda u, v: smoothstep(0.1, -0.5, v)
        else:
            gum_of = lambda u, v: 1 - smoothstep(0.25, 0.55, v)

        def colour(P, N, u, v):
            lt = self.lt or self.sc.lt
            Pl = self.local(P)
            if kind == "cavity":
                # looking into the nose: darkness, a little warmer and lighter toward the lit rim
                d = smoothstep(0.6, 1.9, Pl[:, 1])[:, None] * smoothstep(-0.5, 0.5, Pl[:, 2])[:, None]
                return np.array([0.050, 0.034, 0.024]) + d * np.array([0.085, 0.060, 0.040])
            x, y = Pl[:, 0], Pl[:, 1]
            shut = np.ones(len(P))
            lit = None
            if kind == "core":
                deep = core_carve(x, y) / ORBIT_DEPTH
                under = 1 - smoothstep(0.2, 0.8, y)
                shut = np.clip(1 - 0.92 * deep - 0.5 * under, 0.04, 1)
                # the brow shuts the window out of the socket: only its lower back wall catches a little
                into = smoothstep(0.2, 0.85, deep)
                low = smoothstep(2.0, 1.0, y) * smoothstep(7.0, 5.6, x)
                lit = lt.light.lit(P, skip) * (1 - into * (0.97 - 0.12 * low))
            elif kind == "jaw":
                shut = 0.75 + 0.25 * (1 - smoothstep(-0.9, -0.2, y)) - 0.60 * smoothstep(-0.1, 0.7, y)
                lit = lt.light.lit(P, skip) * (1 - 0.75 * smoothstep(0.0, 0.8, y))        # the blade inside the arch is in its shade
            elif tooth:
                shut = 0.45 + 0.35 * smoothstep(0.18, 0.5, np.abs(y + 0.05))
            if f.inside:
                shut = shut * 0.5
            alb = _albedo(Pl, kind, gum_of(u, v) if tooth else None)
            # bone lets a little light into itself: where it turns from the window the half-tone runs warm
            turn = np.exp(-((normalize(N) @ lt.light.L - 0.02) / 0.20) ** 2)[:, None]
            alb = alb * (1 + turn * np.array([0.10, -0.03, -0.16]))
            return lt.shade(alb, P, N, skip=skip, ao=shut, lit=lit, spec=0.42 if tooth else 0.075,
                            gloss=55 if tooth else 16, crowd=0.8, wrap=0.14)
        return colour

    softness = (0.10, 0.0)

    @property
    def centre(self):
        return self.world([5.0, 1.4, 0.0])

    reach = 8.0

    def groups(self):
        return [(self.sc.cam.depth(self.world([9.6, JAW_FLOOR, 0.6])), self.forms)]     # judged by its nearest point

    def register(self, studio):
        studio.softness_of["skull"] = self.softness
        studio.hook("finish", self.details)

    # ---- the small things that make it bone
    def details(self, studio):
        """Seams where the plates of the skull knit together (fine, in their proper places), the
        openings nerves ran through, the scrolled bones inside the nose."""
        sc = self.sc
        cv, cam, rng, px = sc.cv, sc.cam, sc.rng, sc.px
        core = studio.form("skull_core")
        surf, region = core.surf, core.region

        def on_core(xs, th):
            P, N = surf(np.asarray(xs, dtype=np.float64), np.asarray(th, dtype=np.float64))
            xy = cam.project(P)
            ok = (np.sum(N * cam.to_eye(P), axis=-1) > 0.10) & region.inside(xy, px(1.0)) & ~nasal_opening(self.local(P))
            return xy, ok

        def seam(xs, th, zig=1.0, tone="#8d7c60", thick=0.35, r=0.5):
            xy, ok = on_core(xs, th)
            n = len(xy)
            d = np.gradient(xy, axis=0)
            nr = np.stack([-d[:, 1], d[:, 0]], 1)
            nr /= np.maximum(np.linalg.norm(nr, axis=1, keepdims=True), 1e-9)
            # a suture is a tight zigzag, tighter and looser by turns
            saw = np.where(np.arange(n) % 2 == 0, 1.0, -1.0) * (0.5 + 0.5 * rng.random(n))
            xy = xy + nr * (px(zig) * saw)[:, None]
            idx = np.nonzero(ok)[0]
            if len(idx) < 4:
                return
            for run in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
                if len(run) < 4:
                    continue
                pr = 0.25 + 0.75 * np.clip(0.55 + 0.6 * np.sin(np.linspace(0, 7, len(run)) + rng.uniform(0, 6)), 0, 1)
                cv.stroke(xy[run], Paint.opaque(tone), Brush(radius=px(r), hardness=0.4, thick=thick, pickup=0.2,
                          press_radius=0.5), pr)

        n = 150
        sp = lambda a, b: np.linspace(a, b, n)
        seam(sp(0.6, 4.25), np.zeros(n) + 0.015, zig=0.9)                                  # along the crown
        seam(4.28 + 0.08 * np.sin(sp(0, 6)), sp(-1.2, 1.25), zig=1.2, thick=0.4)           # across it: frontal to parietal
        seam(sp(4.3, 8.05), np.zeros(n) - 0.01, zig=0.5, thick=0.28)                       # down the forehead
        seam(sp(8.05, 9.38), np.zeros(n) + 0.02, zig=0.35, thick=0.33)                     # between the nasal bones
        for sgn in (1, -1):
            seam(sp(7.95, 9.40), sgn * sp(0.50, 0.34), zig=0.4, thick=0.30)                # outer edge of each nasal bone
            seam(8.0 + 0.10 * np.sin(sp(0, 3)), sgn * sp(0.06, 0.50), zig=0.5, thick=0.3)  # nasal to frontal
            seam(sp(9.72, 9.52), sgn * sp(0.75, 1.75), zig=0.4, thick=0.3)                 # premaxilla to maxilla
            seam(sp(4.2, 1.5), sgn * (1.28 + 0.22 * np.sin(sp(0, np.pi))), zig=0.8, thick=0.3)    # the temple
        cv.dry_now()

        # fine pores and specks over the lit bone, each almost nothing
        xy, ok = on_core(rng.uniform(0.5, 9.3, 500), rng.uniform(-1.3, 1.6, 500))
        speck = Batch(cv, Brush(radius=px(0.5), hardness=0.5, thick=0.3, pickup=0.2))
        for q in xy[ok]:
            speck.add(np.array([q, q + rng.normal(0, px(0.5), 2)]), None, 1.0, rng.uniform(0.6, 1.4),
                      paint=Paint.opaque("#84735a"), tscale=rng.uniform(0.3, 1.0))
        speck.flush()

        # the opening below the eye socket, above the second cheek tooth, and two on the side of the chin
        def hole(P_local, w, h, ang=0.0):
            c = cam.project(self.world(P_local))
            d = np.array([np.cos(ang), np.sin(ang)])
            cv.stroke(np.array([c - d * px(w * 0.4), c + d * px(w * 0.4)]), Paint.opaque("#20160e"),
                      Brush(radius=px(h), hardness=0.25, thick=1.0, pickup=0.2), [0.8, 1.0])
        Pc, _ = self.core_local(np.array([8.42]), np.array([1.95]))
        hole(Pc[0], 1.6, 1.3, 0.6)
        for u, v in ((0.12, -0.05), (0.19, -0.2)):
            Pj, _ = self.jaw_local(np.array([u]), np.array([v]))
            hole(Pj[0], 0.9, 0.7, 0.1)

        # inside the nose: the thin scrolled bones, catching a little light in the dark
        for z0, bend in ((0.24, 0.09), (-0.20, -0.07), (0.04, 0.02)):
            pts = np.array([[9.50, 1.60, z0 * 1.3], [9.62, 1.30, z0 + bend], [9.76, 1.02, z0 * 0.7], [9.86, 0.86, z0 * 0.5 - bend]])
            xy = cam.project(self.world(catmull(pts, 0.03)))
            cv.stroke(xy, Paint.opaque("#6f5d46"), Brush(radius=px(0.8), hardness=0.4, thick=0.45, pickup=0.2, press_radius=0.6),
                      np.sin(np.linspace(0.2, np.pi - 0.2, len(xy))))
        cv.dry_now()
