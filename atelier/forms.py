"""Simple solids as parametric surfaces: surf(u, v) -> (points, outward normals)."""
import numpy as np

from .geom import normalize, frame


def sphere(C, R):
    """u: longitude 0..2pi, v: latitude -pi/2..pi/2."""
    C = np.asarray(C, dtype=np.float64)

    def surf(u, v):
        n = np.stack([np.cos(v) * np.cos(u), np.sin(v), np.cos(v) * np.sin(u)], -1)
        return C + R * n, n
    return surf


def ellipsoid(C, radii, rot=None):
    """u: longitude, v: latitude, in the solid's own frame (rot maps local to world)."""
    C = np.asarray(C, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    rot = np.eye(3) if rot is None else np.asarray(rot, dtype=np.float64)

    def surf(u, v):
        s = np.stack([np.cos(v) * np.cos(u), np.sin(v), np.cos(v) * np.sin(u)], -1)
        P = C + (s * radii) @ rot.T
        N = normalize((s / radii) @ rot.T)
        return P, N
    return surf


def plane(origin, eu, ev, flip=False):
    """P = origin + u*eu + v*ev."""
    origin = np.asarray(origin, dtype=np.float64)
    eu = np.asarray(eu, dtype=np.float64)
    ev = np.asarray(ev, dtype=np.float64)
    n = normalize(np.cross(eu, ev)) * (-1 if flip else 1)

    def surf(u, v):
        P = origin + np.asarray(u)[..., None] * eu + np.asarray(v)[..., None] * ev
        return P, np.broadcast_to(n, P.shape)
    return surf


def disc(center, normal, R):
    """u: angle 0..2pi, v: radius 0..1."""
    center = np.asarray(center, dtype=np.float64)
    n = normalize(normal)
    e1, e2 = frame(n)

    def surf(u, v):
        u = np.asarray(u)[..., None]
        v = np.asarray(v)[..., None]
        P = center + R * v * (np.cos(u) * e1 + np.sin(u) * e2)
        return P, np.broadcast_to(n, P.shape)
    return surf


def revolution(base, axis, radius, d_radius=None, ref=None):
    """A turned form. u: angle 0..2pi, v: distance along the axis from `base`.

    radius(v) gives the profile; d_radius(v) its slope (estimated if not given).
    """
    base = np.asarray(base, dtype=np.float64)
    a = normalize(axis)
    e1, e2 = frame(a)
    if ref is not None:                       # choose where angle 0 points
        e1 = normalize(np.asarray(ref, dtype=np.float64) - (np.asarray(ref) @ a) * a)
        e2 = np.cross(a, e1)

    def surf(u, v):
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        r = radius(v)
        dr = d_radius(v) if d_radius is not None else (radius(v + 1e-3) - radius(v - 1e-3)) / 2e-3
        rad = np.cos(u)[..., None] * e1 + np.sin(u)[..., None] * e2
        P = base + v[..., None] * a + r[..., None] * rad
        N = normalize(rad - dr[..., None] * a)
        return P, N
    return surf


class Profile:
    """A turned profile through (height, radius) points, smooth in between."""

    def __init__(self, pts):
        pts = np.asarray(pts, dtype=np.float64)
        self.h, self.r = pts[:, 0], pts[:, 1]
        # monotone-ish smooth interpolation: cubic Hermite with finite-difference tangents
        m = np.gradient(self.r, self.h)
        self.m = m

    def __call__(self, v):
        v = np.clip(np.asarray(v, dtype=np.float64), self.h[0], self.h[-1])
        i = np.clip(np.searchsorted(self.h, v) - 1, 0, len(self.h) - 2)
        h0, h1 = self.h[i], self.h[i + 1]
        t = (v - h0) / (h1 - h0)
        d = h1 - h0
        t2, t3 = t * t, t * t * t
        return ((2 * t3 - 3 * t2 + 1) * self.r[i] + (t3 - 2 * t2 + t) * d * self.m[i]
                + (-2 * t3 + 3 * t2) * self.r[i + 1] + (t3 - t2) * d * self.m[i + 1])


def hash3(xi, yi, zi, seed=0):
    """Repeatable pseudo-random numbers in 0..1 from integer lattice coordinates."""
    h = (xi.astype(np.int64) * 374761393 + yi.astype(np.int64) * 668265263
         + zi.astype(np.int64) * 2147483647 + (int(seed) * 2654435761) % 4294967296) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / float(0x1000000)


def noise3(P, freq=1.0, seed=0):
    """Smooth repeatable noise in 0..1 at points P (..., 3): the same spot always gives the same value."""
    Q = np.asarray(P, dtype=np.float64) * freq
    i = np.floor(Q).astype(np.int64)
    f = Q - i
    f = f * f * (3 - 2 * f)
    out = 0
    for dx in (0, 1):
        wx = f[..., 0] if dx else 1 - f[..., 0]
        for dy in (0, 1):
            wy = f[..., 1] if dy else 1 - f[..., 1]
            for dz in (0, 1):
                wz = f[..., 2] if dz else 1 - f[..., 2]
                out = out + wx * wy * wz * hash3(i[..., 0] + dx, i[..., 1] + dy, i[..., 2] + dz, seed)
    return out


def fbm3(P, freq=1.0, octaves=4, seed=0, gain=0.5):
    """Layered noise, roughly in 0..1."""
    out, amp, tot = 0, 1.0, 0
    for o in range(octaves):
        out = out + amp * noise3(P, freq * 2 ** o, seed + 17 * o)
        tot += amp
        amp *= gain
    return out / tot


# --------------------------------------------------------------------------
# Free forms, for things that are not simple solids (bone, mostly).
# --------------------------------------------------------------------------

def spow(a, e):
    """Signed power: keeps the sign of a."""
    return np.sign(a) * np.abs(a) ** e


class Table1D:
    """Values given at stations along a form, read off anywhere in between."""

    def __init__(self, x, smooth=True, **cols):
        self.x = np.asarray(x, dtype=np.float64)
        self.cols = {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}
        self.smooth = smooth
        self._prof = {k: Profile(np.stack([self.x, v], 1)) for k, v in self.cols.items()} if smooth else None

    def __call__(self, name, x):
        if self.smooth:
            return self._prof[name](x)
        return np.interp(x, self.x, self.cols[name])


def loft(stations, ey=0.9, ez=0.85, carve=None, shape=None):
    """A body built from cross-sections, like a boat hull.

    stations: Table1D over x with columns yc (centre height), w (half width),
    ht / hb (half heights above / below the centre), and optionally ey / ez
    (how square the section is: 1 = elliptical, smaller = boxier).
    carve(x, y) -> 0..1 hollows the sides: the half width is scaled by (1 - carve).
    shape(x, y, z, yc) -> (y, z) has the last word on each cross-section.
    u: position along the body; v: angle round the section (0 = top, pi/2 = +z side).
    """
    has_e = "ez" in stations.cols
    has_b = "eb" in stations.cols

    def pos(u, v):
        c, s = np.cos(v), np.sin(v)
        h = np.where(c > 0, stations("ht", u), stations("hb", u))
        e_y = stations("ey", u) if has_e else ey
        e_z = stations("ez", u) if has_e else ez
        if has_b:                       # a squarer lower half where asked for
            eb = stations("eb", u)
            e_y = np.where(c > 0, e_y, eb)
            e_z = np.where(c > 0, e_z, eb)
        x = np.asarray(u, dtype=np.float64) + 0 * v
        y = stations("yc", u) + h * spow(c, e_y)
        z = stations("w", u) * spow(s, e_z)
        if carve is not None:
            z = z * (1 - carve(x, y))
        if shape is not None:               # shape(x, y, z, yc) -> (y, z): crests, ridges, swellings
            y, z = shape(x, y, z, stations("yc", u) + 0 * v)
        return np.stack([x, y, z], -1)

    def surf(u, v):
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        P = pos(u, v)
        du, dv = 2e-3, 2e-3
        Pu = pos(u + du, v) - pos(u - du, v)
        Pv = pos(u, v + dv) - pos(u, v - dv)
        N = normalize(np.cross(Pu, Pv))
        axis = np.stack([u + 0 * v, stations("yc", u) + 0 * v, 0 * u + 0 * v], -1)
        sign = np.sign(np.sum(N * (P - axis), axis=-1, keepdims=True))
        return P, N * np.where(sign == 0, 1, sign)
    return surf


class Curve:
    """A space curve through points, read by a parameter 0..1 (linear between dense samples)."""

    def __init__(self, pts, smooth=True, n=200, param=None):
        """param: optional parameter value at each given point (so two curves can be kept in step)."""
        from .geom import catmull
        pts = np.asarray(pts, dtype=np.float64)
        if smooth and len(pts) > 2 and param is None:
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
            pts = catmull(pts, step=seg / n)
        self.p = pts
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.length = d.sum()
        if param is None:
            self.s = np.concatenate([[0], np.cumsum(d)]) / self.length
        else:
            q = np.asarray(param, dtype=np.float64)
            self.s = (q - q[0]) / (q[-1] - q[0])

    def __call__(self, t):
        t = np.clip(np.asarray(t, dtype=np.float64), 0, 1)
        return np.stack([np.interp(t, self.s, self.p[:, k]) for k in range(self.p.shape[1])], -1)

    def tangent(self, t, h=2e-3):
        return normalize(self(np.asarray(t) + h) - self(np.asarray(t) - h))


def rails(top, bottom, outward, bulge=0.15, edge=0.6):
    """A strap of bone between two edge curves (Curve objects sharing a parameter).

    u: along the strap 0..1; v: across it, -1 (bottom edge) .. +1 (top edge).
    outward: a rough direction the visible face points in. bulge may be a number or f(u).
    """
    outward = np.asarray(outward, dtype=np.float64)
    bf = bulge if callable(bulge) else (lambda u: np.full_like(np.asarray(u, dtype=np.float64), bulge))

    def surf(u, v):
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        T, B = top(u), bottom(u)
        mid, half = 0.5 * (T + B), 0.5 * (T - B)
        h = np.linalg.norm(half, axis=-1, keepdims=True)
        Wd = half / np.maximum(h, 1e-9)
        tang = normalize(0.5 * (top.tangent(u) + bottom.tangent(u)))
        O = normalize(np.cross(tang, Wd))
        O = O * np.sign(np.sum(O * outward, axis=-1, keepdims=True))
        b = bf(u)[..., None]
        vv = v[..., None]
        P = mid + half * vv + O * b * (1 - vv ** 2)
        N = normalize(h * O + (2 * b * vv + edge * h * vv ** 5) * Wd)
        return P, N
    return surf


def tube(curve, radius, flatten=None):
    """A rod or tapering spike round a Curve. u: angle 0..2pi; v: along 0..1. radius: f(v).

    flatten=(direction, factor) squashes the section along `direction` (for blade-like teeth).
    """
    # one steady frame for the whole curve is enough for the gently bent things painted here
    t0 = curve.tangent(np.array([0.5]))[0]
    if flatten is not None:
        d = np.asarray(flatten[0], dtype=np.float64)
        e2 = normalize(d - (d @ t0) * t0)
        e1 = np.cross(e2, t0)
        fz = flatten[1]
    else:
        e1, e2 = frame(t0)
        fz = 1.0

    def surf(u, v):
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        C = curve(v)
        T = curve.tangent(v)
        a1 = normalize(e1 - np.sum(e1 * T, axis=-1, keepdims=True) * T)
        a2 = np.cross(T, a1)
        r = radius(v)[..., None]
        dr = ((radius(v + 1e-3) - radius(v - 1e-3)) / 2e-3 / max(curve.length, 1e-9))[..., None]
        cu, su = np.cos(u)[..., None], np.sin(u)[..., None]
        P = C + r * (cu * a1 + fz * su * a2)
        N = normalize(cu * a1 * fz + su * a2 - dr * T * fz)
        return P, N
    return surf


def transformed(surf, M, t):
    """Place a form built in its own frame into the world: P -> t + M P (M orthonormal)."""
    M = np.asarray(M, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    def out(u, v):
        P, N = surf(u, v)
        return t + P @ M.T, N @ M.T
    return out


def toward_eye(surf, cam):
    """Both faces of a thin shell: normals are turned to face the eye, so the inside of a
    form (the far half of a jaw seen through the mouth, say) can be painted like any surface."""
    def out(u, v):
        P, N = surf(u, v)
        s = np.sign(np.sum(N * cam.to_eye(P), axis=-1, keepdims=True))
        return P, N * np.where(s == 0, 1, s)
    return out


def roughened(surf, amount, scale=1.0, seed=0):
    """A form that is not quite true: pushed in and out along its normals by slow, repeatable
    unevenness, the way anything made by hand or grown is. amount in cm; scale: lumps per cm."""
    def height(P):
        return amount * 2.0 * (fbm3(P * scale, 1.0, 2, 300 + seed) - 0.5)

    def out(u, v):
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        P, N = surf(u, v)
        P2 = P + N * height(P)[..., None]
        du = 1e-3 * (np.ptp(u) + 1e-3) if np.ndim(u) else 1e-3
        dv = 1e-3 * (np.ptp(v) + 1e-3) if np.ndim(v) else 1e-3
        du, dv = max(du, 1e-4), max(dv, 1e-4)
        Pa, Na = surf(u + du, v)
        Pb, Nb = surf(u, v + dv)
        Pa = Pa + Na * height(Pa)[..., None]
        Pb = Pb + Nb * height(Pb)[..., None]
        n = np.cross(Pa - P2, Pb - P2)
        n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
        n = n * np.sign(np.sum(n * N, axis=-1, keepdims=True) + 1e-12)
        return P2, n
    return out

