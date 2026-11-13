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
