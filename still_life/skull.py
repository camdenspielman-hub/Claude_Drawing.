"""The cat's skull, modelled like clay from its anatomy and checked against photographs of real ones.

Own frame (cm): X runs from the back of the head (0) to the front teeth (10), Y is up from the
biting edge of the upper tooth row, Z is the animal's left. What makes it a cat and not a dog or
a fox: the face in front of the eye sockets is barely a fifth of the length; the sockets are huge,
face forward, and stay open behind (two spurs of bone reach toward each other and never meet);
the cheekbone arches flare far wider than the braincase; the top is one arc; the lower jaw is two
slim bars, each with a tall blade rising inside its arch.

The skull is one sculpted solid (atelier/solid.py), so bone runs into bone the way it grows, the
hollows are real hollows, and it throws its own shadows into them.
"""
import numpy as np

from atelier.brushwork import Batch
from atelier.canvas import Brush, Paint
from atelier.forms import fbm3, noise3
from atelier.geom import Ball, Capsule, Glow, normalize, smoothstep, catmull
from atelier.shading import lin
from atelier.solid import Solid
from atelier.stencil import Stencil
from atelier.studio import Form
from . import scene as L

BONE, UPPER_TEETH, JAW, LOWER_TEETH = 0, 1, 2, 3
JAW_FLOOR = -1.31                     # the lowest edge of the lower jaw: it rests on this

# the cheekbone arch: X, Y, Z of its middle line and half its height
_ARCH = np.array([
    [8.15, 1.05, 1.35, 0.26], [7.75, 1.08, 1.85, 0.40], [7.20, 1.10, 2.45, 0.44], [6.50, 1.14, 3.00, 0.44],
    [5.75, 1.20, 3.38, 0.42], [4.95, 1.26, 3.52, 0.36], [4.15, 1.30, 3.42, 0.30], [3.45, 1.32, 3.10, 0.27],
    [2.95, 1.32, 2.65, 0.27], [2.60, 1.30, 2.15, 0.28], [2.30, 1.28, 1.60, 0.28]])
# the lower jaw's bar
_BAR = np.array([[9.78, -0.72, 0.10], [9.45, -0.80, 0.55], [8.70, -0.84, 1.00], [7.80, -0.85, 1.38],
                 [6.90, -0.84, 1.72], [6.00, -0.78, 2.00], [5.20, -0.66, 2.18]])


def sculpt(s):
    """Model the skull in the solid s."""
    m = dict(mirror=True)
    # ---- the braincase, forehead and muzzle: one flowing body, domed, the face short and steep
    s.lump([2.60, 2.30, 0], [2.55, 1.90, 2.30])
    s.lump([5.40, 2.65, 0], [1.90, 1.42, 1.25], k=0.5)
    s.rod([6.50, 3.05, 0], [9.25, 1.50, 0], 0.88, 0.48, k=0.4)              # down the bridge of the nose
    s.lump([8.40, 1.15, 0], [1.60, 0.95, 1.35], k=0.4)
    s.lump([7.85, 0.95, 1.20], [0.90, 0.60, 0.70], k=0.35, **m)              # the cheek in front of the eye
    s.lump([8.10, 0.55, 0], [1.90, 0.30, 1.30], k=0.3)                       # the palate
    s.chain([[9.86, 0.42, 0.35], [9.50, 0.45, 1.05], [8.60, 0.50, 1.40], [7.60, 0.52, 1.75], [6.75, 0.52, 1.95]],
            0.36, k=0.3, **m)                                                # the ridge the upper teeth are set in
    s.lump([3.00, 0.80, 0], [2.60, 0.50, 1.70], k=0.4)                       # the floor of the braincase
    s.lump([2.30, 0.50, 1.15], [0.80, 0.50, 0.60], k=0.2, **m)               # the ear capsules
    s.lump([0.55, 0.88, 0.50], [0.28, 0.22, 0.26], k=0.35, **m)              # where the neck joined
    s.lump([1.35, 1.15, 1.40], [0.45, 0.40, 0.30], k=0.35, **m)
    s.rod([0.50, 3.75, 0], [2.60, 4.14, 0], 0.09, 0.05, k=0.18)              # the low crest along the crown
    s.rod([9.20, 1.35, 1.12], [9.38, 0.45, 1.20], 0.30, 0.30, k=0.25, **m)   # the swelling over the fang's root
    # ---- the hollows
    s.lump([4.30, 2.00, 2.70], [1.50, 1.35, 1.00], k=0.5, carve=True, **m)   # the temple
    s.lump([6.55, 1.95, 2.05], [1.35, 1.30, 1.35], k=0.06, carve=True, **m)  # the eye socket, sharp-rimmed
    s.lump([6.50, 1.55, 1.75], [1.25, 1.00, 1.32], k=0.15, carve=True, **m)  # deepening under the brow
    s.rod([10.40, 1.32, 0], [8.90, 1.50, 0], 0.60, 0.42, k=0.1, carve=True)  # the bony nostril, running back into the nose
    s.lump([9.80, 1.58, 0.22], [0.50, 0.40, 0.34], k=0.1, carve=True, **m)   # wider above: a heart, not a hole
    s.lump([8.30, 1.25, 1.60], [0.13, 0.17, 0.25], k=0.05, carve=True, **m)  # the opening under the eye
    s.lump([0.02, 1.05, 0], [0.36, 0.33, 0.36], k=0.08, carve=True)          # where the spinal cord left
    # ---- the spurs behind the eye, and the cheekbone arch (a strap: deep, and half as thick)
    s.rod([5.75, 3.10, 1.00], [5.45, 2.75, 1.95], 0.32, 0.10, k=0.2, **m)
    s.chain(_ARCH[:, :3], _ARCH[:, 3] + 0.10, k=0.10, flat=(2, 0.42), **m)
    s.rod([5.80, 1.50, 3.40], [5.62, 2.10, 3.28], 0.26, 0.07, k=0.2, **m)
    s.lump([2.70, 1.30, 2.20], [0.70, 0.30, 0.55], k=0.25, **m)              # the shelf the arch springs from
    # ---- upper teeth
    t = dict(group=UPPER_TEETH, mirror=True)
    s.chain([[9.30, 0.60, 1.14], [9.44, -0.15, 1.22], [9.40, -0.92, 1.24]], [0.25, 0.19, 0.02], k=0.05, **t)
    for z in (0.12, 0.33):
        s.rod([10.05, 0.45, z], [10.10, 0.05, z], 0.085, 0.07, **t)
    s.rod([10.00, 0.45, 0.57], [10.04, -0.05, 0.60], 0.12, 0.07, **t)
    s.rod([8.95, 0.40, 1.33], [8.95, 0.12, 1.34], 0.09, 0.05, **t)
    s.rod([8.35, 0.50, 1.50], [8.35, -0.14, 1.54], 0.24, 0.03, **t)
    s.rod([8.60, 0.50, 1.44], [8.60, 0.15, 1.46], 0.15, 0.04, k=0.10, **t)
    s.rod([8.10, 0.50, 1.58], [8.10, 0.12, 1.60], 0.15, 0.04, k=0.10, **t)
    s.rod([7.85, 0.52, 1.68], [7.85, 0.08, 1.70], 0.17, 0.04, **t)           # the shearing tooth: three points
    s.rod([7.50, 0.55, 1.78], [7.50, -0.27, 1.82], 0.25, 0.03, k=0.10, **t)
    s.rod([7.05, 0.55, 1.92], [7.00, -0.12, 1.95], 0.23, 0.05, k=0.12, **t)
    s.lump([6.72, 0.28, 1.92], [0.12, 0.14, 0.16], **t)
    # ---- the lower jaw
    j = dict(group=JAW, mirror=True)
    for sgn in (1, -1):
        s.chain(_BAR + np.array([0, sgn * 0.22, 0]), 0.25, k=0.2, **j)
    s.lump([9.72, -0.72, 0], [0.30, 0.45, 0.30], k=0.2, group=JAW)           # the chin, where the halves join
    s.rod([5.50, -0.45, 2.15], [4.10, 1.95, 2.40], 0.80, 0.20, k=0.3, flat=(2, 0.30), **j)   # the blade inside the arch
    s.rod([4.40, 0.00, 2.30], [3.15, 0.40, 2.30], 0.45, 0.20, k=0.3, flat=(2, 0.35), **j)
    s.rod([3.10, 0.42, 1.85], [3.10, 0.42, 2.60], 0.17, 0.17, k=0.15, **j)   # the hinge
    s.rod([4.60, -0.65, 2.25], [3.25, -0.60, 2.30], 0.35, 0.14, k=0.3, flat=(2, 0.40), **j)
    s.lump([4.70, 0.35, 2.62], [0.80, 0.85, 0.20], k=0.2, carve=True, **j)
    # ---- lower teeth
    t = dict(group=LOWER_TEETH, mirror=True)
    s.chain([[9.50, -0.55, 0.62], [9.70, 0.00, 0.80], [9.68, 0.50, 0.88]], [0.20, 0.16, 0.02], k=0.05, **t)
    for z in (0.10, 0.25, 0.40):
        s.rod([9.92, -0.50, z], [9.98, -0.14, z], 0.06, 0.06, **t)
    s.rod([8.75, -0.50, 1.00], [8.75, -0.02, 1.03], 0.20, 0.03, **t)
    s.rod([8.20, -0.50, 1.22], [8.20, 0.02, 1.25], 0.22, 0.03, **t)
    s.rod([7.98, -0.50, 1.30], [7.98, -0.25, 1.32], 0.14, 0.04, k=0.08, **t)
    s.rod([7.55, -0.50, 1.45], [7.60, 0.05, 1.47], 0.23, 0.04, **t)
    s.rod([7.15, -0.50, 1.60], [7.10, 0.12, 1.62], 0.24, 0.04, k=0.10, **t)


BONE_COL = lin("#cdc3a8")
BONE_STAIN = lin("#9c7f4d")
BONE_THIN = lin("#bdb6a6")
ENAMEL = lin("#e2dbc6")


class Skull:
    softness = (0.10, 0.0)
    reach = 8.0

    def __init__(self, sc, pos=None, yaw=None):
        self.sc = sc
        pos = L.SKULL if pos is None else pos
        yaw = L.SKULL_YAW if yaw is None else yaw
        fwd = np.array([-np.cos(yaw), 0.0, np.sin(yaw)])
        up = np.array([0.0, 1.0, 0.0])
        left = np.cross(up, fwd)
        self.k = L.SKULL_SCALE
        self.M0 = np.stack([fwd, up, left], axis=1)
        self.t = np.asarray(pos, dtype=np.float64) - (self.M0 * self.k) @ np.array([5.0, JAW_FLOOR, 0.0])
        self.solid = Solid(self.M0, self.t, self.k, centre=(5.0, 1.3, 0.0), radius=6.3,
                           rough=(0.018, 1.4), grain=(0.003, 11.0), seed=5)
        sculpt(self.solid)
        self.lt = None
        self.axis = fwd
        self._build()

    # ---- frames
    def world(self, P_local):
        return self.solid.world(P_local)

    def local(self, P_world):
        return self.solid.local(P_world)

    @property
    def centre(self):
        return self.world([5.0, 1.4, 0.0])

    def hides(self, P):
        """True for points that lie behind some part of the skull, as we look at them."""
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        eye = self.sc.cam.eye
        d = P - eye
        dist = np.linalg.norm(d, axis=1)
        t, _ = self.solid.trace(np.broadcast_to(eye, P.shape), d / dist[:, None])
        return ~np.isnan(t) & (t < dist - 0.06)

    # ---- what it blocks of the light for its neighbours, and what it throws back at them
    def occluders(self):
        k = self.k
        occ = [Ball("skull:core", self.world([2.5, 2.1, 0]), k, scale=(2.5, 1.8, 2.3), rot=self.M0),
               Ball("skull:core", self.world([5.5, 2.5, 0]), k, scale=(1.9, 1.3, 1.6), rot=self.M0),
               Ball("skull:core", self.world([8.3, 1.3, 0]), k, scale=(1.75, 1.0, 1.4), rot=self.M0),
               Ball("under:skull", self.world([5.6, 0.2, 0]), k, scale=(4.3, 1.1, 3.3), rot=self.M0)]
        for sgn, s in ((1, "L"), (-1, "R")):
            m = np.array([1, 1, sgn])
            pts = _ARCH[::2, :3] * m
            for a, b in zip(pts[:-1], pts[1:]):
                occ.append(Capsule("skull:arch_" + s, self.world(a), self.world(b), 0.36 * k))
            bar = _BAR[::2] * m
            for a, b in zip(bar[:-1], bar[1:]):
                occ.append(Capsule("skull:jaw_" + s, self.world(a), self.world(b), 0.46 * k))
            occ.append(Ball("skull:jaw_" + s, self.world(np.array([4.7, 0.3, 2.3]) * m), k, scale=(1.1, 1.3, 0.25), rot=self.M0))
        return occ

    def glows(self):
        return [Glow("skull", self.world([3.2, 2.0, 0]), 2.4 * self.k, BONE_COL),
                Glow("skull", self.world([7.6, 1.6, 0]), 1.6 * self.k, BONE_COL)]

    # ---- the forms: bone, lower jaw, teeth, each where it actually shows
    def _build(self):
        sc = self.sc
        seen = self.solid.seen_from(sc.cam)
        self.seen = seen
        part = seen.outline(sc.W, sc.H)
        self.part = part

        def flow(P, N, which):
            around = normalize(np.cross(N, self.axis))
            return around if which == "u" else normalize(np.cross(N, around))

        def of(groups):
            return lambda P, N, u, v: np.isin(self.solid.group(P), groups)

        def add(name, groups, size, **kw):
            f = Form("skull_" + name, seen, (0, sc.W), (0, sc.H), self._colour, owner="skull", size=size,
                     stencil=Stencil(np.isin(part, groups)), flow=flow, vis=of(groups), along="v", **kw)
            self.forms.append(f)
            return f
        self.forms = []
        add("bone", [BONE], 5.0, tol=0.028)
        add("jaw", [JAW], 3.6, tol=0.028)
        add("teeth", [UPPER_TEETH, LOWER_TEETH], 1.3, dead=False, tol=0.03)

    def albedo(self, Pl, g, open_small, open_big):
        """Local colour of old cleaned bone: warm ivory, greyer where it is thin, browner where grease
        and dust have lodged in every hollow and seam; enamel whiter and glassier."""
        tooth = (g == UPPER_TEETH) | (g == LOWER_TEETH)
        broad = smoothstep(0.42, 0.78, fbm3(Pl * 0.75, 1, 3, 11))[:, None]
        fine = fbm3(Pl * 6.5, 1, 3, 23)[:, None]
        pores = noise3(Pl * 34.0, 1.0, 31)[:, None]
        a = BONE_COL * (1 - 0.34 * broad) + BONE_STAIN * 0.34 * broad
        thin = (smoothstep(0.3, 1.5, Pl[:, 1] - 2.2) * smoothstep(5.2, 3.4, Pl[:, 0]))[:, None]
        a = a * (1 - 0.30 * thin) + BONE_THIN * 0.30 * thin
        lodged = (smoothstep(0.92, 0.45, open_small) * 0.55 + smoothstep(0.85, 0.3, open_big) * 0.35)[:, None]
        a = a * (1 - lodged) + BONE_STAIN * 0.62 * lodged
        a = np.where((g == JAW)[:, None], a * np.array([1.0, 0.955, 0.87]), a)
        a = a * (0.90 + 0.16 * fine) * (0.95 + 0.10 * pores)
        # teeth: the crown pale and a little blue-grey toward the tip, yellow-brown at the gum line
        root = np.where(g == UPPER_TEETH, smoothstep(0.05, 0.5, Pl[:, 1]), smoothstep(-0.12, -0.5, Pl[:, 1]))[:, None]
        e = ENAMEL * (0.95 + 0.05 * fine)
        e = e * (1 - 0.42 * root * np.array([0.30, 0.55, 1.0])) * (1 - 0.06 * (1 - root) * np.array([1.0, 0.7, 0.3]))
        return np.where(tooth[:, None], e, a), tooth

    def _colour(self, P, N, u, v):
        lt = self.lt or self.sc.lt
        s = self.solid
        skip = ("skull", "under:skull")
        Lw = lt.light.L
        g = s.group(P)
        Pl = s.local(P)
        open_small = s.openness(P, N, reach=0.45 * self.k)
        open_big = s.openness(P, N, reach=2.6 * self.k)
        alb, tooth = self.albedo(Pl, g, open_small, open_big)
        # bone lets a little light into itself: where it turns from the window the half-tone runs warm
        turn = np.exp(-((N @ Lw - 0.02) / 0.22) ** 2)[:, None] * (~tooth)[:, None]
        alb = alb * (1 + turn * np.array([0.10, -0.03, -0.16]))
        lit = lt.light.lit(P, skip) * s.sees(P, N, Lw, reach=16.0, soft=6.0)
        shut = np.clip(open_small, 0, 1) ** 0.8 * (0.12 + 0.88 * np.clip(open_big, 0, 1) ** 1.6)
        # the eye socket and the nostril are the darkest things in the picture: the brow and the
        # nasal bones keep the window out of them, whatever a stray ray might do
        q = np.linalg.norm((np.abs(Pl * [1, 1, 1]) * 0 + (Pl * np.array([1, 1, 0]) + np.abs(Pl) * np.array([0, 0, 1]))
                            - np.array([6.55, 1.95, 2.05])) / np.array([1.35, 1.30, 1.35]), axis=1)
        socket = smoothstep(1.14, 0.98, q) * smoothstep(1.95, 1.45, np.abs(Pl[:, 2]))
        nostril = (smoothstep(9.86, 9.60, Pl[:, 0]) * smoothstep(0.62, 0.50, np.hypot(Pl[:, 1] - 1.42, Pl[:, 2]))
                   * smoothstep(8.6, 9.0, Pl[:, 0]))
        hollow = np.maximum(socket, nostril)
        lit = lit * (1 - 0.86 * hollow)
        shut = shut * (1 - 0.62 * hollow)
        out = np.empty((len(P), 3))
        for sel, spec, gloss in ((~tooth, 0.085, 18.0), (tooth, 0.45, 60.0)):
            if sel.any():
                out[sel] = lt.shade(alb[sel], P[sel], N[sel], skip=skip, ao=shut[sel], lit=lit[sel], spec=spec,
                                    gloss=gloss, crowd=0.8, wrap=0.14)
        return out

    def groups(self):
        return [(self.sc.cam.depth(self.world([9.6, JAW_FLOOR, 0.6])), self.forms)]     # judged by its nearest point

    def register(self, studio):
        studio.softness_of["skull"] = self.softness
        studio.hook("finish", self.details)

    # ---- the small things that make it bone
    def details(self, studio):
        """Seams where the plates of the skull knit together (fine, in their proper places), the
        little openings nerves ran through, the scrolled bones inside the nose."""
        sc = self.sc
        cv, cam, rng, px = sc.cv, sc.cam, sc.rng, sc.px
        region = studio.regions["skull"]
        s = self.solid

        def on_bone(guide):
            """A guide line (own frame) laid onto the surface; returns canvas points and which are seen."""
            P = s.onto(self.world(catmull(np.asarray(guide, dtype=np.float64), 0.035)))
            xy = cam.project(P)
            ok = ~self.hides(P) & region.inside(xy, px(1.0)) & (np.sum(s.normal(P) * cam.to_eye(P), axis=1) > 0.12)
            return xy, ok

        def seam(guide, zig=1.0, tone="#8a7657", thick=0.35, r=0.5):
            xy, ok = on_bone(guide)
            n = len(xy)
            if n < 6:
                return
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

        seam([[0.7, 3.8, 0.02], [2.0, 4.0, 0.0], [3.2, 4.0, 0.03], [4.3, 3.9, 0.0]], zig=0.9)             # along the crown
        seam([[4.35, 2.9, -1.5], [4.3, 3.6, -0.8], [4.4, 3.9, 0.0], [4.3, 3.6, 0.8], [4.35, 2.9, 1.5], [4.5, 2.2, 1.9]],
             zig=1.2, thick=0.4)                                                                         # across it
        seam([[4.4, 3.9, 0.0], [5.6, 3.8, 0.02], [6.8, 3.5, 0.0], [7.6, 3.1, 0.0]], zig=0.5, thick=0.28)  # down the forehead
        seam([[7.6, 3.1, 0.0], [8.4, 2.6, 0.01], [9.3, 2.1, 0.0]], zig=0.35, thick=0.33)                  # between the nasal bones
        for sg in (1, -1):
            seam([[7.6, 3.0, 0.1 * sg], [8.3, 2.5, 0.45 * sg], [9.3, 1.95, 0.42 * sg]], zig=0.4, thick=0.30)
            seam([[9.6, 0.9, 0.75 * sg], [9.3, 1.3, 1.25 * sg], [9.0, 1.8, 0.95 * sg]], zig=0.4, thick=0.3)
            seam([[4.3, 2.4, 2.0 * sg], [3.4, 2.75, 2.25 * sg], [2.3, 2.6, 2.4 * sg], [1.5, 2.0, 2.2 * sg]], zig=0.8, thick=0.3)
            seam([[5.0, 1.15, 3.6 * sg], [4.6, 0.85, 3.6 * sg], [4.3, 0.7, 3.5 * sg]], zig=0.5, thick=0.3)  # across the arch
            seam([[7.9, 1.3, 1.9 * sg], [7.6, 0.9, 2.2 * sg], [7.4, 0.55, 2.3 * sg]], zig=0.5, thick=0.3)
        cv.dry_now()

        # fine pores and specks over the bone, each almost nothing
        ys, xs = np.nonzero(studio.form("skull_bone").region.mask | studio.form("skull_jaw").region.mask)
        if len(xs):
            pick = rng.integers(0, len(xs), int(520 * sc.s ** 0 ))
            speck = Batch(cv, Brush(radius=px(0.5), hardness=0.5, thick=0.3, pickup=0.2))
            for i in pick:
                q = np.array([xs[i] + rng.random(), ys[i] + rng.random()])
                speck.add(np.array([q, q + rng.normal(0, px(0.5), 2)]), None, 1.0, rng.uniform(0.6, 1.4),
                          paint=Paint.opaque("#7f6c50"), tscale=rng.uniform(0.3, 1.0))
            speck.flush()

        # two little openings on the side of the chin
        def hole(P_local, w, h, ang=0.0):
            P = s.onto(self.world(np.array([P_local])))
            if self.hides(P)[0]:
                return
            c = cam.project(P)[0]
            d = np.array([np.cos(ang), np.sin(ang)])
            cv.stroke(np.array([c - d * px(w * 0.4), c + d * px(w * 0.4)]), Paint.opaque("#20160e"),
                      Brush(radius=px(h), hardness=0.25, thick=1.0, pickup=0.2), [0.8, 1.0])
        hole([9.2, -0.85, 0.9], 0.9, 0.7, 0.1)
        hole([8.5, -0.9, 1.3], 0.8, 0.6, 0.1)

        # inside the nose: the thin scrolled bones, catching a little light in the dark
        for z0, bend in ((0.24, 0.09), (-0.20, -0.07), (0.04, 0.02)):
            pts = np.array([[9.42, 1.72, z0 * 1.3], [9.50, 1.42, z0 + bend], [9.58, 1.14, z0 * 0.7], [9.64, 0.94, z0 * 0.5 - bend]])
            P = self.world(catmull(pts, 0.03))
            ok = ~self.hides(P)
            if ok.sum() < 4:
                continue
            xy = cam.project(P[ok])
            cv.stroke(xy, Paint.opaque("#6f5d46"), Brush(radius=px(0.8), hardness=0.4, thick=0.45, pickup=0.2, press_radius=0.6),
                      np.sin(np.linspace(0.2, np.pi - 0.2, len(xy))))
        cv.dry_now()


STUDIO_READY = True
