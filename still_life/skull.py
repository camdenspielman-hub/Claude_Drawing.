"""The cat's skull, constructed from anatomy.

Built in its own frame (cm): X runs from the back of the head (0) to the front
teeth (10), Y is up from the plane of the upper tooth row, Z is the animal's
left. Measurements follow an adult domestic cat: about 10 cm long, 6.8 cm
across the cheekbones, a short blunt muzzle, very large forward-facing orbits
and a round braincase.
"""
import numpy as np

from atelier import forms
from atelier.brushwork import paint_surface, line
from atelier.canvas import Brush, Paint
from atelier.forms import Table1D, Curve, fbm3
from atelier.geom import Ball, Capsule, normalize, smoothstep, catmull
from atelier.shading import lin
from atelier.stencil import Stencil
from . import scene as L

TWO_PI = 2 * np.pi
MIRROR = np.diag([1.0, 1.0, -1.0])

# ---- the central body: braincase, brow and muzzle, as cross-sections from back to front
CORE = Table1D(
    [0.0, 0.3, 1.0, 2.0, 3.0, 4.0, 5.0, 5.6, 6.2, 6.8, 7.4, 8.0, 8.6, 9.1, 9.5, 9.85, 10.05],
    w=[0.60, 1.55, 2.08, 2.25, 2.25, 2.05, 1.72, 1.85, 2.00, 1.75, 1.60, 1.45, 1.36, 1.32, 1.24, 1.04, 0.74],
    yc=[1.55, 1.55, 1.70, 1.80, 1.85, 1.90, 1.98, 1.98, 1.92, 1.80, 1.60, 1.35, 1.15, 1.02, 0.90, 0.70, 0.55],
    ht=[0.60, 1.35, 1.85, 2.12, 2.15, 2.10, 1.93, 1.82, 1.70, 1.56, 1.42, 1.20, 0.95, 0.80, 0.65, 0.42, 0.25],
    hb=[0.70, 1.15, 1.40, 1.50, 1.50, 1.50, 1.50, 1.50, 1.48, 1.42, 1.28, 1.07, 0.89, 0.77, 0.65, 0.45, 0.25],
    ez=[0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.80, 0.70, 0.62, 0.58, 0.58, 0.60, 0.70, 0.85],
    ey=[0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.88, 0.84, 0.80, 0.80, 0.80, 0.80, 0.85, 0.90],
    # underneath, the muzzle is squared off: a flat palate with upright sides down to the tooth row
    eb=[0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.80, 0.62, 0.50, 0.45, 0.42, 0.42, 0.42, 0.45, 0.60, 0.85])

ORBIT_C = (6.45, 1.90)     # centre of the eye socket on the side of the head (X, Y)
ORBIT_R = (1.25, 1.30)


def core_carve(x, y):
    """The hollows in the side of the head: the eye socket, and the shallower temple behind it."""
    r = np.sqrt(((x - ORBIT_C[0]) / ORBIT_R[0]) ** 2 + ((y - ORBIT_C[1]) / ORBIT_R[1]) ** 2)
    orbit = 0.52 * (1 - smoothstep(0.62, 1.0, r))
    rt = np.sqrt(((x - 4.6) / 1.4) ** 2 + ((y - 1.3) / 1.1) ** 2)
    temple = 0.16 * (1 - smoothstep(0.3, 1.0, rt))
    return np.maximum(orbit, temple)


def core_top(x):
    return CORE("yc", x) + CORE("ht", x)


def nasal_opening(P_local):
    """True where the front of the muzzle is open (the bony nostril)."""
    x, y, z = P_local[..., 0], P_local[..., 1], P_local[..., 2]
    rim = 1.82 - 1.55 * (x - 9.1)                        # slanting from the nasal tips down to the incisors
    half = 0.18 + 0.40 * np.clip((y - 0.6) / 1.0, 0, 1)  # pear-shaped: wide above, narrow below
    return (x > 9.1) & (y > rim) & (np.abs(z) < half)


# ---- cheekbone arch (left side): top and bottom edges
_ARCH = np.array([
    # X     Ytop  Ybot  Ztop  Zbot
    [8.30, 0.85, 0.60, 1.20, 1.25],     # grows out of the cheek
    [7.85, 1.20, 0.30, 1.48, 1.60],     # the deep plate under the front of the eye socket
    [7.30, 1.02, 0.28, 2.05, 2.15],
    [6.75, 0.92, 0.32, 2.60, 2.68],
    [6.42, 0.95, 0.36, 2.88, 2.94],
    [6.05, 2.20, 0.42, 2.74, 3.10],     # the upward spur toward the brow
    [5.66, 1.05, 0.48, 3.22, 3.26],
    [5.00, 0.96, 0.56, 3.40, 3.40],     # a slim bar from here back
    [4.30, 0.98, 0.62, 3.40, 3.40],
    [3.70, 1.02, 0.66, 3.18, 3.18],
    [3.20, 1.06, 0.66, 2.85, 2.85],
    [2.85, 1.08, 0.62, 2.50, 2.50],
    [2.50, 0.95, 0.75, 2.05, 2.05]])    # fades into the braincase above the jaw joint
ARCH_TOP = Curve(_ARCH[:, [0, 1, 3]], smooth=False)
ARCH_BOT = Curve(_ARCH[:, [0, 2, 4]], smooth=False)
ARCH_MID = Curve(np.stack([_ARCH[:, 0], 0.5 * (np.minimum(_ARCH[:, 1], 1.2) + _ARCH[:, 2]), _ARCH[:, 4]], 1))

# ---- the ridge of bone along the upper tooth row, from the front teeth round to behind the carnassial
_RIDGE = np.array([
    # X      Z     Ytop  Ybot
    [10.08, 0.00, 0.46, 0.22],
    [10.06, 0.36, 0.46, 0.22],
    [9.99, 0.68, 0.48, 0.22],
    [9.80, 1.00, 0.56, 0.22],
    [9.40, 1.30, 0.86, 0.20],     # swelling over the root of the canine
    [9.05, 1.40, 0.90, 0.20],
    [8.55, 1.46, 0.80, 0.24],
    [8.05, 1.60, 0.76, 0.26],
    [7.55, 1.78, 0.74, 0.27],
    [7.05, 1.95, 0.72, 0.28],
    [6.55, 2.08, 0.66, 0.30],
    [6.30, 2.06, 0.56, 0.36]])
RIDGE_TOP = Curve(np.stack([_RIDGE[:, 0], _RIDGE[:, 2], _RIDGE[:, 1] - 0.30], 1), param=np.arange(len(_RIDGE)))
RIDGE_BOT = Curve(np.stack([_RIDGE[:, 0], _RIDGE[:, 3], _RIDGE[:, 1] + 0.02], 1), param=np.arange(len(_RIDGE)))

# ---- lower jaw (left half): top and bottom edges
_JAW = np.array([
    # X     Ytop   Ybot   Z
    [9.90, -0.36, -0.80, 0.00],     # the chin, where the two halves join
    [9.84, -0.36, -0.86, 0.22],
    [9.45, -0.34, -1.28, 0.55],
    [8.80, -0.36, -1.46, 1.02],
    [8.00, -0.40, -1.52, 1.38],
    [7.00, -0.44, -1.56, 1.70],
    [6.20, -0.36, -1.58, 1.93],
    [5.60, 0.30, -1.60, 2.08],
    [5.00, 0.95, -1.56, 2.20],
    [4.50, 1.38, -1.44, 2.28],
    [4.10, 1.52, -1.32, 2.33],    # tip of the coronoid process
    [3.65, 1.00, -1.18, 2.36],
    [3.25, 0.45, -1.06, 2.36],
    [2.90, 0.40, -0.98, 2.33]])
JAW_TOP = Curve(_JAW[:, [0, 1, 3]], smooth=False)
JAW_BOT = Curve(_JAW[:, [0, 2, 3]])


def _jaw_bulge(u):
    return np.interp(u, [0, 0.45, 0.6, 0.8, 1.0], [0.20, 0.26, 0.10, -0.05, 0.08])


def _spike(a, m, b, r0, power=0.75, flatten=None):
    """A canine: a stout tapering tooth from root a through m to tip b."""
    return forms.tube(Curve([a, m, b]), lambda v: r0 * np.clip(1 - v, 0, 1) ** power + 0.012, flatten=flatten)


def _peg(a, b, r0):
    """An incisor: a tiny peg with a rounded end."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return forms.tube(Curve([a, 0.5 * (a + b), b], smooth=False),
                      lambda v: r0 * np.sqrt(np.clip(1 - np.clip(v, 0, 1) ** 3, 0, 1)) + 0.008)


def _crown(xs, gum_y, tip_y, z0, z1, upper=True, thick=0.11):
    """A cheek tooth: one crown between its gum line and the jagged profile of its cusps.

    xs: stations along the jaw; gum_y / tip_y: heights of the gum line and of the cutting edge there.
    """
    xs = np.asarray(xs, dtype=np.float64)
    zs = z0 + (xs - xs[0]) / (xs[-1] - xs[0]) * (z1 - z0)
    gum = Curve(np.stack([xs, np.broadcast_to(gum_y, xs.shape), zs], 1), smooth=False, param=xs)
    tip = Curve(np.stack([xs, np.asarray(tip_y, dtype=np.float64), zs + 0.03], 1), smooth=False, param=xs)
    return forms.rails(gum, tip, [0, 0, 1], bulge=thick, edge=0.8) if upper else forms.rails(tip, gum, [0, 0, 1], bulge=thick, edge=0.8)


class Part:
    def __init__(self, name, surf, ur, vr, kind="bone", side=0, grid=64):
        self.name, self.surf, self.ur, self.vr = name, surf, ur, vr
        self.kind, self.side, self.grid = kind, side, grid
        self.inside = False


class Skull:
    """All the parts, placed on the table, in far-to-near painting order."""

    def __init__(self, sc, pos=None, yaw=None):
        self.sc = sc
        pos = L.SKULL if pos is None else pos
        yaw = L.SKULL_YAW if yaw is None else yaw
        fwd = np.array([-np.cos(yaw), 0.0, np.sin(yaw)])
        up = np.array([0.0, 1.0, 0.0])
        left = np.cross(up, fwd)
        self.M = np.stack([fwd, up, left], axis=1)
        self.t = np.asarray(pos, dtype=np.float64) - self.M @ np.array([5.0, -1.60, 0.0])
        self.parts = []
        self._build()

    # ---- frames
    def world(self, P_local):
        return self.t + np.asarray(P_local, dtype=np.float64) @ self.M.T

    def local(self, P_world):
        return (np.asarray(P_world, dtype=np.float64) - self.t) @ self.M

    def place(self, surf, mirror=False):
        M = self.M @ MIRROR if mirror else self.M
        return forms.transformed(surf, M, self.t)

    def add(self, name, surf, ur, vr, kind="bone", mirror=False, grid=64, both=False):
        placed = self.place(surf, bool(mirror))
        if both:
            placed = forms.toward_eye(placed, self.sc.cam)
        p = Part(name + ("_R" if mirror else "_L" if mirror is not None else ""), placed, ur, vr, kind, grid=grid)
        p.inside = both
        self.parts.append(p)

    def _build(self):
        core = forms.loft(CORE, carve=core_carve)
        brow_spur = forms.tube(Curve([[6.05, 3.10, 1.20], [6.03, 2.98, 1.60], [6.00, 2.78, 1.98]], smooth=False),
                               lambda v: 0.34 * np.clip(1 - v, 0, 1) ** 0.6 + 0.02, flatten=([0.1, 0.8, 0.6], 0.45))
        arch = forms.rails(ARCH_TOP, ARCH_BOT, [0, 0, 1], bulge=0.10, edge=0.6)
        jaw = forms.rails(JAW_TOP, JAW_BOT, [0, 0, 1], bulge=_jaw_bulge, edge=0.9)
        bulla = forms.ellipsoid([1.95, 0.0, 1.25], (0.95, 0.72, 0.68))
        ridge = forms.rails(RIDGE_TOP, RIDGE_BOT, [0.35, 0, 1], bulge=0.025, edge=0.25)
        up_canine = _spike([9.10, 0.62, 1.08], [9.30, -0.28, 1.19], [9.27, -0.88, 1.21], 0.26, power=0.62)
        lo_canine = _spike([9.48, -0.78, 0.72], [9.68, 0.02, 0.84], [9.64, 0.42, 0.90], 0.21, power=0.62)
        # six small front teeth above and six below, set in an arc along the front of each jaw
        teeth_up = [_peg([x - 0.06, 0.52, z], [x - 0.01, tip, z], r)
                    for x, z, r, tip in [(10.07, 0.11, 0.095, 0.02), (10.05, 0.31, 0.10, 0.00), (10.00, 0.55, 0.13, -0.08)]]
        teeth_lo = [_peg([x - 0.08, -0.66, z], [x, -0.12, z], 0.075)
                    for x, z in [(9.90, 0.08), (9.88, 0.23), (9.84, 0.38)]]
        # upper cheek teeth: a tiny peg, a single-pointed premolar, and the long three-pointed carnassial.
        # Their crowns start well up behind the ridge of bone, so they are seen growing out of it.
        cheek_up = [_crown([8.52, 8.46, 8.40], 0.50, [0.22, 0.12, 0.22], 1.34, 1.37, thick=0.06),
                    _crown([8.24, 8.14, 8.03, 7.95, 7.87, 7.78, 7.68], 0.55, [0.24, 0.12, -0.08, -0.15, -0.08, 0.10, 0.22],
                           1.44, 1.60, thick=0.17),
                    _crown([7.64, 7.55, 7.48, 7.40, 7.30, 7.18, 7.10, 7.02, 6.94, 6.84, 6.72, 6.60, 6.48], 0.58,
                           [0.22, 0.06, 0.00, 0.06, -0.10, -0.21, -0.23, -0.18, 0.00, -0.08, -0.10, 0.02, 0.18],
                           1.66, 1.96, thick=0.19)]
        # lower cheek teeth sit just inside the upper ones and are rooted below the jaw's top edge
        cheek_lo = [_crown([8.56, 8.44, 8.34, 8.24, 8.12], -0.66, [-0.34, -0.26, -0.22, -0.26, -0.34], 1.06, 1.22, upper=False, thick=0.13),
                    _crown([8.05, 7.92, 7.82, 7.76, 7.70, 7.60, 7.50], -0.68, [-0.36, -0.28, -0.16, -0.13, -0.16, -0.28, -0.36],
                           1.22, 1.40, upper=False, thick=0.14),
                    _crown([7.42, 7.30, 7.22, 7.14, 7.00, 6.86, 6.78, 6.70, 6.50], -0.70,
                           [-0.36, -0.20, -0.12, -0.16, -0.26, -0.16, -0.12, -0.18, -0.34], 1.40, 1.62, upper=False, thick=0.14)]

        def lower(mirror):
            self.add("bulla", bulla, (0, TWO_PI), (-np.pi / 2, np.pi / 2), mirror=mirror, grid=32)
            for i, t in enumerate(cheek_lo):
                self.add("cheeklo%d" % i, t, (0, 1), (-1, 1), kind="tooth", mirror=mirror, grid=36, both=mirror)
            for i, t in enumerate(teeth_lo):
                self.add("inclo%d" % i, t, (0, TWO_PI), (0, 1), kind="tooth", mirror=mirror, grid=16)
            self.add("canine_lo", lo_canine, (0, TWO_PI), (0, 1), kind="tooth", mirror=mirror, grid=28)
            # the jaw covers their roots. The far half is seen from inside, through the mouth.
            self.add("jaw", jaw, (0, 1), (-1, 1), kind="jaw", mirror=mirror, grid=90, both=mirror)

        def upper(mirror):
            for i, t in enumerate(cheek_up):
                self.add("cheekup%d" % i, t, (0, 1), (-1, 1), kind="tooth", mirror=mirror, grid=36, both=mirror)
            for i, t in enumerate(teeth_up):
                self.add("incup%d" % i, t, (0, TWO_PI), (0, 1), kind="tooth", mirror=mirror, grid=16)
            self.add("canine_up", up_canine, (0, TWO_PI), (0, 1), kind="tooth", mirror=mirror, grid=28)

        # far side first, then the middle, then the side that faces us
        lower(True)
        self.add("arch", arch, (0, 1), (-1, 1), mirror=True, grid=90)
        upper(True)
        self.add("ridge", ridge, (0, 1), (-1, 1), mirror=True, grid=70, both=True)        # the far gum, inside and out
        self.parts.append(Part("nose", self.place(core), (8.9, 10.05), (-1.4, 1.4), "cavity", grid=60))
        self.parts.append(Part("core", self.place(core), (0.0, 10.05), (-np.pi, np.pi), "core", grid=140))
        self.add("brow", brow_spur, (0, TWO_PI), (0, 1), mirror=True, grid=24)
        self.add("brow", brow_spur, (0, TWO_PI), (0, 1), mirror=False, grid=24)
        lower(False)
        upper(False)
        self.add("ridge", ridge, (0, 1), (-1, 1), mirror=False, grid=70)                  # covers the near roots
        self.add("arch", arch, (0, 1), (-1, 1), mirror=False, grid=90)

    def part(self, name):
        return next(p for p in self.parts if p.name == name)

    # ---- what it blocks of the light (simple stand-in solids, named so parts can ignore themselves)
    def occluders(self):
        occ = [Ball("skull:core", self.world([2.6, 1.9, 0]), 1.0, scale=(2.65, 2.1, 2.2), rot=self.M),
               Ball("skull:core", self.world([6.1, 2.75, 0]), 1.0, scale=(1.7, 0.95, 1.85), rot=self.M),
               Ball("skull:core", self.world([6.4, 1.3, 0]), 1.0, scale=(1.6, 1.0, 0.85), rot=self.M),
               Ball("skull:core", self.world([8.6, 1.35, 0]), 1.0, scale=(1.5, 1.1, 1.38), rot=self.M)]
        for sgn, s in ((1, "L"), (-1, "R")):
            m = np.array([1, 1, sgn])
            occ.append(Capsule("skull:brow_" + s, self.world(np.array([6.03, 3.0, 1.5]) * m), self.world(np.array([6.0, 2.82, 1.9]) * m), 0.14))
            pts = ARCH_MID(np.linspace(0, 1, 7)) * m
            for a, b in zip(pts[:-1], pts[1:]):
                occ.append(Capsule("skull:arch_" + s, self.world(a), self.world(b), 0.42))
            occ.append(Capsule("skull:arch_" + s, self.world(np.array([6.05, 1.3, 3.05]) * m), self.world(np.array([6.05, 2.05, 2.85]) * m), 0.2))
            jm = 0.5 * (JAW_TOP(np.linspace(0, 0.45, 5)) + JAW_BOT(np.linspace(0, 0.45, 5))) * m  # the body of the jaw
            for a, b in zip(jm[:-1], jm[1:]):
                occ.append(Capsule("skull:jaw_" + s, self.world(a), self.world(b), 0.55))
            occ.append(Ball("skull:jaw_" + s, self.world(np.array([4.6, -0.6, 2.2]) * m), 1.0, scale=(1.7, 1.3, 0.3), rot=self.M))
            occ.append(Capsule("skull:canine_up_" + s, self.world(np.array([9.15, 0.2, 1.12]) * m), self.world(np.array([9.28, -0.75, 1.2]) * m), 0.16))
            occ.append(Ball("skull:bulla_" + s, self.world(np.array([1.95, 0.0, 1.25]) * m), 0.75))
        return occ


# ---------------------------------------------------------------------------
# Painting
# ---------------------------------------------------------------------------
BONE = lin("#cbc2aa")
BONE_STAIN = lin("#b0945f")
JAW_BONE = lin("#c0a56e")
ENAMEL = lin("#ddd6c2")


def _albedo(Pl, kind, gum=None):
    """Local colour of the bone: ivory, with broad warm staining and fine mottling."""
    if kind == "tooth":
        # enamel: off-white, a little grey and glassy toward the point, stained warm at the gum
        g = np.zeros(len(Pl)) if gum is None else gum
        tint = ENAMEL * (0.94 + 0.06 * fbm3(Pl * 9, 1, 2, 5))[:, None]
        tint = tint * (1 - 0.30 * g[:, None] * np.array([0.25, 0.55, 1.0]))
        return tint * (1 - 0.10 * (1 - g[:, None]) * np.array([1.0, 0.8, 0.5]))
    base = JAW_BONE if kind == "jaw" else BONE
    broad = smoothstep(0.42, 0.72, fbm3(Pl * 0.85, 1, 3, 11))[:, None]
    fine = fbm3(Pl * 6.5, 1, 3, 23)[:, None]
    grain = fbm3(Pl * [22, 22, 22], 1, 2, 31)[:, None]
    return (base * (1 - 0.55 * broad) + BONE_STAIN * 0.55 * broad) * (0.86 + 0.20 * fine) * (0.95 + 0.10 * grain)


def _hollow(Pl, kind):
    """How open to the room's light a spot is: 1 in the open, less inside the socket and under the arch."""
    x, y, z = Pl[:, 0], Pl[:, 1], np.abs(Pl[:, 2])
    if kind == "core":
        deep = core_carve(x, y) / 0.52
        under = 1 - smoothstep(0.2, 0.8, y)
        return np.clip(1 - 0.60 * deep - 0.5 * under, 0.2, 1)
    if kind == "jaw":
        return 0.72 + 0.28 * (1 - smoothstep(-0.9, -0.2, y)) - 0.42 * smoothstep(-0.1, 0.7, y)
    if kind == "tooth":
        # teeth are open to the room at their points and shut in near the gum
        return 0.45 + 0.35 * smoothstep(0.18, 0.5, np.abs(y + 0.05))
    return np.ones(len(x))


def _vis(self, p):
    if p.name == "core":
        return lambda P, N, u, v: ~nasal_opening(self.local(P))
    if p.name == "nose":
        return lambda P, N, u, v: nasal_opening(self.local(P))
    return None


def _prepare(self):
    """Draw the outline of every part; their union is the skull's place on the canvas."""
    sc = self.sc
    self.stencils = {}
    self.stencil = Stencil.empty(sc.W, sc.H)
    for p in self.parts:
        st = Stencil.from_surface(sc.cam, p.surf, p.ur, p.vr, sc.W, sc.H, grid=p.grid, visible=_vis(self, p))
        self.stencils[p.name] = st
        self.stencil = self.stencil | st
    return self.stencil


def _ground_ao(self, P):
    """How much the skull shuts out the room's light from the table beneath and beside it."""
    Pl = self.local(P)
    d = np.sqrt(((Pl[:, 0] - 6.3) / 4.2) ** 2 + (Pl[:, 2] / 2.6) ** 2)
    return 1 - 0.75 * (1 - smoothstep(0.55, 1.25, d))


def _paint(self):
    sc = self.sc
    cv, cam, lt, rng, px = sc.cv, sc.cam, sc.lt, sc.rng, sc.px
    if not hasattr(self, "stencils"):
        self.prepare()
    for p in self.parts:
        base = p.name.rsplit("_", 1)[0] if p.name[-2:] in ("_L", "_R") else p.name
        tag = p.name[-1] if p.name[-2:] in ("_L", "_R") else ""
        vis = _vis(self, p)
        st = self.stencils[p.name]
        if not st.any():
            continue

        if p.kind == "cavity":
            # looking into the nose: darkness, warmer and a little lighter toward the lit rim
            def col(P, N, u, v):
                Pl = self.local(P)
                d = smoothstep(0.5, 2.2, Pl[:, 1])[:, None] * smoothstep(0.0, 0.7, Pl[:, 2] + 0.35)[:, None]
                return np.array([0.060, 0.040, 0.028]) + d * np.array([0.10, 0.07, 0.045])
            for r in (2.6, 1.3):
                paint_surface(cv, cam, p.surf, col, Brush(radius=px(r), hardness=0.5, thick=1.5, pickup=0.5), rng,
                              urange=p.ur, vrange=p.vr, length=px(4 * r), spacing=px(0.6 * r), stencil=st, visible=vis,
                              margin=px(0.5), grid=40, min_facing=-2.0, coverage=1.6)
            cv.dry_now()
            continue

        skip = {"core": ("skull:core",), "brow": ("skull:core", "skull:brow_" + tag), "arch": ("skull:arch_" + tag,),
                "ridge": ("skull:core", "skull:canine_up_" + tag),
                "jaw": ("skull:jaw_" + tag,), "bulla": ("skull:bulla_" + tag,),
                "canine_up": ("skull:canine_up_" + tag,)}.get(base, ())
        tooth = p.kind == "tooth"

        if base.startswith("cheekup"):
            gum_of = lambda u, v: smoothstep(-0.1, 0.5, v)
        elif base.startswith("cheeklo"):
            gum_of = lambda u, v: smoothstep(0.1, -0.5, v)
        else:
            gum_of = lambda u, v: 1 - smoothstep(0.25, 0.55, v)

        def col(P, N, u, v, p=p, skip=skip, tooth=tooth, gum_of=gum_of):
            Pl = self.local(P)
            alb = _albedo(Pl, p.kind, gum_of(u, v) if tooth else None)
            return lt.shade(alb, P, N, skip=skip, ao=_hollow(Pl, p.kind) * (0.55 if p.inside else 1.0),
                            spec=0.30 if tooth else 0.06, gloss=45 if tooth else 12, wrap=0.10)

        small = tooth and base not in ("canine_up", "canine_lo")
        along = "v" if p.name in ("core",) else "u"
        if tooth:
            along = "v"
        r1 = px(1.3 if small else 2.2 if tooth else 5.0)
        # lay-in
        paint_surface(cv, cam, p.surf, col, Brush(radius=r1, hardness=0.5, thick=1.4, pickup=0.5), rng,
                      urange=p.ur, vrange=p.vr, along=along, length=r1 * 4.5, spacing=r1 * 0.7, stencil=st,
                      visible=vis, grid=max(20, p.grid // 2), coverage=3.0 if small else 1.2,
                      margin=px(0.5) if small else None)
        if not small:
            # modelling: a smaller brush, working wet into the lay-in and across it
            r2 = px(1.3 if tooth else 2.3)
            paint_surface(cv, cam, p.surf, col, Brush(radius=r2, hardness=0.55, thick=1.2, pickup=0.55), rng,
                          urange=p.ur, vrange=p.vr, along="u" if along == "v" else "v", length=r2 * 5,
                          spacing=r2 * 0.75, stencil=st, margin=r2 * 0.5, visible=vis, grid=max(20, p.grid // 2))
            # shadow edges and other quick changes: the point of a small brush, short touches
            r4 = px(1.2)
            paint_surface(cv, cam, p.surf, col, Brush(radius=r4, hardness=0.55, thick=1.2, pickup=0.6), rng,
                          urange=p.ur, vrange=p.vr, along=along, length=r4 * 3.2, spacing=r4 * 0.8, stencil=st,
                          margin=r4 * 0.5, visible=vis, grid=max(20, p.grid // 2), detail=(0.03, px(3.5)))
            # soften with a clean soft brush, following the form
            paint_surface(cv, cam, p.surf, col, Brush(kind="blender", radius=px(3.0 if not tooth else 1.6), hardness=0.2,
                                                      pickup=0.30, load=0.5), rng,
                          urange=p.ur, vrange=p.vr, along=along, length=px(16), spacing=px(4.0), stencil=st,
                          visible=vis, grid=max(20, p.grid // 2), paint=Paint.opaque("#000000"))
        # the edge, with the point of a small brush
        r3 = px(1.0)
        paint_surface(cv, cam, p.surf, col, Brush(radius=r3, hardness=0.7, thick=1.2, pickup=0.6), rng,
                      urange=p.ur, vrange=p.vr, along=along, length=px(6), spacing=px(1.1), stencil=st, margin=px(0.5),
                      coverage=2.0 if small else 1.0,
                      visible=lambda P, N, u, v, st=st, vis=vis: (st.depth(cam.project(P)) < px(3.2)) & (True if vis is None else vis(P, N, u, v)),
                      grid=max(20, p.grid // 2))
        cv.dry_now()


Skull.paint = _paint
Skull.prepare = _prepare
Skull.ground_ao = _ground_ao
