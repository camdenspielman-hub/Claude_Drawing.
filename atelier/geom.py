"""Draughtsman's geometry: perspective, curves, and constructed shadows."""
import numpy as np


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


class Camera:
    """Pinhole perspective. World: x right, y up, z toward the viewer (cm)."""

    def __init__(self, eye, target, focal, W, H, cx=None, cy=None):
        self.eye = np.asarray(eye, dtype=np.float64)
        self.f = normalize(np.asarray(target, dtype=np.float64) - self.eye)
        self.r = normalize(np.cross(self.f, [0.0, 1.0, 0.0]))
        self.u = np.cross(self.r, self.f)
        self.focal = float(focal)
        self.W, self.H = W, H
        self.cx = W / 2 if cx is None else cx
        self.cy = H / 2 if cy is None else cy

    def project(self, P):
        """World points (..., 3) -> canvas pixels (..., 2)."""
        d = np.asarray(P, dtype=np.float64) - self.eye
        z = d @ self.f
        x = self.cx + self.focal * (d @ self.r) / z
        y = self.cy - self.focal * (d @ self.u) / z
        return np.stack([x, y], axis=-1)

    def depth(self, P):
        return (np.asarray(P, dtype=np.float64) - self.eye) @ self.f

    def onto_plane(self, xy, origin, normal):
        """World points where the lines of sight through canvas points xy (N, 2) meet a plane."""
        xy = np.asarray(xy, dtype=np.float64)
        d = (self.f[None, :] + ((xy[:, 0] - self.cx) / self.focal)[:, None] * self.r[None, :]
             - ((xy[:, 1] - self.cy) / self.focal)[:, None] * self.u[None, :])
        n = np.asarray(normal, dtype=np.float64)
        t = ((np.asarray(origin, dtype=np.float64) - self.eye) @ n) / (d @ n)
        return self.eye + t[:, None] * d

    def to_eye(self, P):
        """Unit vectors from the points toward the eye."""
        return normalize(self.eye - np.asarray(P, dtype=np.float64))

    def px_per_cm(self, P):
        return self.focal / self.depth(P)


def resample(pts, step):
    """Re-space a polyline (N, d) at about `step` along its length."""
    pts = np.asarray(pts, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return pts[:1]
    n = max(2, int(np.ceil(s[-1] / step)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, k]) for k in range(pts.shape[1])], axis=1)


def catmull(ctrl, step=4.0, closed=False, per_seg=16):
    """A smooth curve through control points (N, d), re-spaced at `step`."""
    P = np.asarray(ctrl, dtype=np.float64)
    if len(P) < 3:
        return resample(P, step)
    if closed:
        P = np.concatenate([P[-1:], P, P[:2]])
    else:
        P = np.concatenate([2 * P[:1] - P[1:2], P, 2 * P[-1:] - P[-2:-1]])
    t = np.linspace(0, 1, per_seg, endpoint=False)[:, None]
    out = []
    for i in range(len(P) - 3):
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2:-1])
    return resample(np.concatenate(out), step)


def frame(normal):
    """Two unit vectors spanning the plane perpendicular to `normal`."""
    n = normalize(normal)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = normalize(np.cross(n, a))
    e2 = np.cross(n, e1)
    return e1, e2


def circle3(center, normal, radius, n=72, a0=0.0, a1=2 * np.pi):
    """Points of a circle (or arc) in space."""
    e1, e2 = frame(normal)
    a = np.linspace(a0, a1, n)[:, None]
    return np.asarray(center, dtype=np.float64) + radius * (np.cos(a) * e1 + np.sin(a) * e2)


def taper(n, head=0.2, tail=0.3, lo=0.0, hi=1.0):
    """Pressure along a stroke: ease in, hold, ease out."""
    t = np.linspace(0, 1, n)
    p = np.minimum(1.0, np.minimum(t / max(head, 1e-6), (1 - t) / max(tail, 1e-6)))
    return lo + (hi - lo) * np.clip(p, 0, 1) ** 0.8


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------
# Constructed shadows. With one distant light, the shadow of a simple solid is
# found the way a draughtsman finds it: follow the light back from the point
# in question and see how close that line passes to the solid. The blur of
# the shadow's edge grows with the distance between the two.
# --------------------------------------------------------------------------

class Occluder:
    def __init__(self, name, opacity=1.0):
        self.name = name
        self.opacity = opacity

    def cover(self, P, L, soft):
        """How much of the light is blocked at points P (N, 3): 0 = none, 1 = all."""
        raise NotImplementedError

    def bound(self):
        """A sphere (centre, radius) that contains it."""
        raise NotImplementedError

    def nearest(self, P):
        """For each point, the closest point of this solid's core and the solid's reach round it."""
        c, r = self.bound()
        return np.broadcast_to(c, P.shape), r

    def shuts_out(self, P, N):
        """How much of the surrounding room this solid hides from a surface at P facing N (0..1):
        the darkening where things touch or crowd together. Treated as a ball at its nearest point."""
        q, r = self.nearest(P)
        v = q - P
        d = np.maximum(np.linalg.norm(v, axis=1), 1e-6)
        facing = np.clip(np.sum(v * N, axis=1) / d, 0, 1)
        return self.opacity * np.clip((r / np.maximum(d, r)) ** 2, 0, 1) * facing


class Ball(Occluder):
    def __init__(self, name, center, radius, opacity=1.0, scale=(1, 1, 1), rot=None):
        """A sphere, or an ellipsoid when `scale`/`rot` are given (rot maps local to world)."""
        super().__init__(name, opacity)
        self.c = np.asarray(center, dtype=np.float64)
        self.r = radius
        self.scale = np.asarray(scale, dtype=np.float64)
        self.rot = np.eye(3) if rot is None else np.asarray(rot, dtype=np.float64)

    def cover(self, P, L, soft):
        # work in the frame where the ellipsoid is a unit-scale sphere
        Pl = ((P - self.c) @ self.rot) / self.scale
        Ll = (L @ self.rot) / self.scale
        ll = np.linalg.norm(Ll)
        Ld = Ll / ll
        t = -(Pl @ Ld)                       # distance along the ray to the closest approach
        closest = Pl + t[:, None] * Ld
        d = np.linalg.norm(closest, axis=1)
        w = np.maximum(soft * np.maximum(t, 0) / ll, 0.04) / np.mean(self.scale)
        c = 1 - smoothstep(self.r - w, self.r + w, d)
        # points on the lit side of the ball itself are not shadowed by it
        return self.opacity * c * (t > 0)

    def bound(self):
        return self.c, self.r * float(np.max(self.scale))

    def nearest(self, P):
        return np.broadcast_to(self.c, P.shape), self.r * float(np.mean(self.scale))


class Capsule(Occluder):
    def __init__(self, name, a, b, radius, opacity=1.0):
        super().__init__(name, opacity)
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.r = radius

    def cover(self, P, L, soft):
        # closest distance between the ray P + sL (s >= 0) and the segment a + t(b - a)
        d2 = self.b - self.a
        r = P - self.a
        e = d2 @ d2
        b = L @ d2
        c = r @ L
        f = r @ d2
        den = e - b * b
        s = (b * f - c * e) / den if abs(den) > 1e-9 else np.zeros(len(P))
        s = np.maximum(s, 0)
        t = np.clip((b * s + f) / e, 0, 1)
        s = np.maximum(t * b - c, 0)
        d = np.linalg.norm(r + s[:, None] * L - t[:, None] * d2, axis=1)
        w = np.maximum(soft * s, 0.03)
        return self.opacity * (1 - smoothstep(self.r - w, self.r + w, d)) * (s > 1e-6)

    def bound(self):
        return 0.5 * (self.a + self.b), 0.5 * np.linalg.norm(self.b - self.a) + self.r

    def nearest(self, P):
        d2 = self.b - self.a
        t = np.clip(((P - self.a) @ d2) / max(d2 @ d2, 1e-12), 0, 1)
        return self.a + t[:, None] * d2, self.r * 1.25


class Disc(Occluder):
    def __init__(self, name, center, normal, radius, opacity=1.0):
        super().__init__(name, opacity)
        self.c = np.asarray(center, dtype=np.float64)
        self.n = normalize(normal)
        self.r = radius

    def cover(self, P, L, soft):
        den = L @ self.n
        if abs(den) < 1e-6:
            return np.zeros(len(P))
        t = ((self.c - P) @ self.n) / den
        hit = P + t[:, None] * L
        d = np.linalg.norm(hit - self.c, axis=1)
        w = np.maximum(soft * np.maximum(t, 0), 0.03)
        return self.opacity * (1 - smoothstep(self.r - w, self.r + w, d)) * (t > 1e-4)

    def bound(self):
        return self.c, self.r

    def nearest(self, P):
        # the closest point of the disc: drop onto its plane, pull in to the rim
        v = P - self.c
        v = v - (v @ self.n)[:, None] * self.n
        d = np.maximum(np.linalg.norm(v, axis=1), 1e-9)
        q = self.c + v * np.minimum(1.0, 0.7 * self.r / d)[:, None]
        return q, self.r * 0.55


class Quad(Occluder):
    """A flat convex polygon (corners in order)."""

    def __init__(self, name, corners, opacity=1.0):
        super().__init__(name, opacity)
        self.p = np.asarray(corners, dtype=np.float64)
        self.n = normalize(np.cross(self.p[1] - self.p[0], self.p[2] - self.p[0]))

    def cover(self, P, L, soft):
        den = L @ self.n
        if abs(den) < 1e-6:
            return np.zeros(len(P))
        t = ((self.p[0] - P) @ self.n) / den
        hit = P + t[:, None] * L
        inside = np.full(len(P), 1e9)
        m = len(self.p)
        for i in range(m):
            a, b = self.p[i], self.p[(i + 1) % m]
            edge_n = normalize(np.cross(self.n, b - a))    # points inward
            inside = np.minimum(inside, (hit - a) @ edge_n)
        w = np.maximum(soft * np.maximum(t, 0), 0.03)
        return self.opacity * smoothstep(-w, w, inside) * (t > 1e-4)

    def bound(self):
        c = self.p.mean(axis=0)
        return c, float(np.max(np.linalg.norm(self.p - c, axis=1)))

    def nearest(self, P):
        c, r = self.bound()
        v = P - c
        v = v - (v @ self.n)[:, None] * self.n
        d = np.maximum(np.linalg.norm(v, axis=1), 1e-9)
        return c + v * np.minimum(1.0, 0.6 * r / d)[:, None], r * 0.4


class Glow:
    """Something bright and coloured enough to throw its colour on its neighbours and to show in
    polished things near it: a ball of a given colour (linear), standing in for the object."""

    def __init__(self, name, center, radius, colour):
        self.name = name
        self.c = np.asarray(center, dtype=np.float64)
        self.r = float(radius)
        self.colour = np.asarray(colour, dtype=np.float64)


class Light:
    """One window: a direction, how soft its shadows are, and what stands in its way."""

    def __init__(self, toward, soft=0.06, occluders=None):
        self.L = normalize(toward)
        self.soft = soft
        self.occluders = [] if occluders is None else occluders

    def add(self, *occ):
        self.occluders.extend(occ)

    def lit(self, P, skip=()):
        """Share of the light reaching points P (N, 3), ignoring occluders named in `skip`."""
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        vis = np.ones(len(P))
        for o in self.occluders:
            if any(o.name.startswith(s) for s in skip):
                continue
            vis *= 1 - o.cover(P, self.L, self.soft)
        return vis

    def open_to_room(self, P, N, skip=(), reach=1.0):
        """Share of the room's dim light reaching a surface at P facing N: less where other
        solids crowd close (under a jaw, between coins, where a ball meets the table)."""
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        keep = np.ones(len(P))
        for o in self.occluders:
            if any(o.name.startswith(s) for s in skip):
                continue
            keep *= 1 - np.clip(reach * o.shuts_out(P, N), 0, 0.95)
        return keep

    def near(self, center, radius, margin=10.0):
        """The same light with only the solids that could shadow or crowd a form of this size here."""
        C = np.asarray(center, dtype=np.float64)
        keep = []
        for o in self.occluders:
            c, r = o.bound()
            v = c - C
            along = v @ self.L
            across = np.linalg.norm(v - along * self.L)
            shadows = along > -(radius + r) and across < radius + r + self.soft * max(along, 0) * 1.5 + 0.5
            crowds = np.linalg.norm(v) < radius + r + margin
            if shadows or crowds:
                keep.append(o)
        return Light(self.L, self.soft, keep)
