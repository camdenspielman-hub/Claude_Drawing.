"""Sculpted solids: forms modelled like clay, for things no ruler-and-compass construction suits.

A Solid is a list of lumps (ellipsoids and tapering rods) that run smoothly into one another,
with hollows pressed into them. It is built in its own frame and then placed in the world.
It answers a draughtsman's questions (see engine/solid.cpp); it never makes an image. A stroke
that is about to be laid asks it: what part of you lies under this spot of the canvas, which
way does it face, does it see the window, how shut in is it?
"""
import ctypes
import os

import numpy as np

from .geom import normalize
from .stencil import Stencil

_HERE = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_HERE, "..", "engine", "libsolid.so"))
_dp = ctypes.POINTER(ctypes.c_double)
_ip = ctypes.POINTER(ctypes.c_int)
_lib.solid_eval.argtypes = [_dp, ctypes.c_int, _dp, _dp, ctypes.c_int, _dp, _ip]
_lib.solid_trace.argtypes = [_dp, ctypes.c_int, _dp, _dp, _dp, ctypes.c_int, _dp, ctypes.c_double, ctypes.c_int,
                             ctypes.c_double, _dp, _dp]
_lib.solid_normal.argtypes = [_dp, ctypes.c_int, _dp, _dp, ctypes.c_int, ctypes.c_double, _dp]
_lib.solid_shadow.argtypes = [_dp, ctypes.c_int, _dp, _dp, ctypes.c_int, _dp, ctypes.c_double, ctypes.c_double, _dp]
_lib.solid_open.argtypes = [_dp, ctypes.c_int, _dp, _dp, _dp, ctypes.c_int, ctypes.c_double, _dp]


def _d(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _p(a):
    return a.ctypes.data_as(_dp)


class Solid:
    def __init__(self, M0, t, k=1.0, centre=(0, 0, 0), radius=10.0, rough=(0.0, 1.0), grain=(0.0, 1.0), seed=1):
        """M0: 3x3 whose columns are the solid's own axes in the world; t: where its origin sits;
        k: its size. centre/radius: a ball (own frame) that holds all of it. rough/grain:
        (height, frequency) of the broad and fine unevenness of its surface."""
        self.M0 = np.asarray(M0, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64)
        self.k = float(k)
        self.centre = _d(centre)
        self.radius = float(radius)
        self.rough = _d([rough[0], rough[1], seed, grain[0], grain[1]])
        self._rows = []
        self._prims = None

    # ---- modelling
    def _add(self, kind, op, group, mirror, k, params):
        row = np.zeros(16)
        row[:5] = [kind, op, group, 1.0 if mirror else 0.0, k]
        row[5:5 + len(params)] = params
        self._rows.append(row)
        self._prims = None

    def lump(self, c, r, k=0.0, group=0, mirror=False, carve=False):
        """An ellipsoid at c with radii r, joined on (or pressed in, if carve) with a fillet of size k."""
        self._add(0, 1 if carve else 0, group, mirror, k, list(c) + list(np.broadcast_to(r, 3)))

    def rod(self, a, b, ra, rb=None, k=0.0, group=0, mirror=False, carve=False, flat=None):
        """A tapering rod from a (radius ra) to b (radius rb); flat=(axis, thinness) squashes it."""
        rb = ra if rb is None else rb
        axis, thin = (-1, 1.0) if flat is None else flat
        self._add(1, 1 if carve else 0, group, mirror, k, list(a) + list(b) + [ra, rb, axis, thin])

    def wound_ball(self, c, squash, R, strand, belts, group=0):
        """A ball of wound strand: belts = [(axis, offset, half_width, layer_height), ...], later
        ones lying over earlier ones (see engine/solid.cpp)."""
        self._add(3, 0, group, False, 0.0, list(c) + list(squash) + [R, strand, len(belts)])
        for axis, off, half, lift in belts:
            self._add(9, 0, group, False, 0.0, list(axis) + [off, half, lift])

    def chain(self, pts, radii, **kw):
        pts = np.asarray(pts, dtype=np.float64)
        radii = np.broadcast_to(np.asarray(radii, dtype=np.float64), (len(pts),))
        for i in range(len(pts) - 1):
            self.rod(pts[i], pts[i + 1], radii[i], radii[i + 1], **kw)

    @property
    def prims(self):
        if self._prims is None:
            self._prims = _d(np.array(self._rows).ravel())
        return self._prims

    # ---- frames
    def local(self, P):
        return (np.asarray(P, dtype=np.float64) - self.t) @ self.M0 / self.k

    def world(self, Pl):
        return self.t + (np.asarray(Pl, dtype=np.float64) * self.k) @ self.M0.T

    # ---- questions (all in world coordinates)
    def distance(self, P, groups=False):
        Pl = _d(self.local(P).reshape(-1, 3))
        d = np.empty(len(Pl))
        g = np.empty(len(Pl), dtype=np.int32)
        _lib.solid_eval(_p(self.prims), len(self._rows), _p(self.rough), _p(Pl), len(Pl), _p(d), g.ctypes.data_as(_ip))
        return (d * self.k, g) if groups else d * self.k

    def group(self, P):
        return self.distance(P, groups=True)[1]

    def trace(self, O, D, steps=110, eps=0.0015):
        """First touch of the lines from O along unit D: distances (nan where they miss), and how near the misses came."""
        Ol = _d(self.local(np.asarray(O, dtype=np.float64).reshape(-1, 3)))
        Dl = _d(np.asarray(D, dtype=np.float64).reshape(-1, 3) @ self.M0)
        n = len(Ol)
        t = np.empty(n)
        near = np.empty(n)
        _lib.solid_trace(_p(self.prims), len(self._rows), _p(self.rough), _p(Ol), _p(Dl), n, _p(self.centre),
                         self.radius, steps, eps, _p(t), _p(near))
        return np.where(t < 0, np.nan, t * self.k), near

    def normal(self, P, e=0.004):
        Pl = _d(self.local(P).reshape(-1, 3))
        N = np.empty_like(Pl)
        _lib.solid_normal(_p(self.prims), len(self._rows), _p(self.rough), _p(Pl), len(Pl), e, _p(N))
        return N @ self.M0.T

    def sees(self, P, N, L, reach=14.0, soft=7.0):
        """0..1: how much of the light from direction L reaches P past the rest of the solid."""
        Pl = _d(self.local(np.asarray(P) + 0.02 * self.k * np.asarray(N)).reshape(-1, 3))
        Ll = _d(np.asarray(L, dtype=np.float64) @ self.M0)
        out = np.empty(len(Pl))
        _lib.solid_shadow(_p(self.prims), len(self._rows), _p(self.rough), _p(Pl), len(Pl), _p(Ll), reach / self.k, soft, _p(out))
        return out

    def openness(self, P, N, reach=1.0):
        """0..1: how open P is to the room (low in hollows and crevices)."""
        Pl = _d(self.local(P).reshape(-1, 3))
        Nl = _d(np.asarray(N, dtype=np.float64).reshape(-1, 3) @ self.M0)
        out = np.empty(len(Pl))
        _lib.solid_open(_p(self.prims), len(self._rows), _p(self.rough), _p(Pl), _p(Nl), len(Pl), reach / self.k, _p(out))
        return out

    def onto(self, P, steps=8):
        """Move points onto the surface along the shortest way (for placing drawn details on it)."""
        P = np.array(P, dtype=np.float64).reshape(-1, 3)
        for _ in range(steps):
            d = self.distance(P)
            P = P - self.normal(P) * d[:, None]
        return P

    # ---- seen from a viewpoint
    def seen_from(self, cam):
        return Seen(self, cam)


class Seen:
    """A solid as it presents itself to one viewpoint. Called like a surface, with canvas
    coordinates for parameters: surf(x, y) -> the point of the solid under that spot of the canvas
    and the way it faces (normal zero where there is nothing)."""
    screen = True

    def __init__(self, solid, cam):
        self.solid, self.cam = solid, cam

    def rays(self, xy):
        cam = self.cam
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        d = (cam.f[None, :] + ((xy[:, 0] - cam.cx) / cam.focal)[:, None] * cam.r[None, :]
             - ((xy[:, 1] - cam.cy) / cam.focal)[:, None] * cam.u[None, :])
        return np.broadcast_to(cam.eye, d.shape), normalize(d)

    def __call__(self, u, v):
        u, v = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
        shape = u.shape
        O, D = self.rays(np.stack([u.ravel(), v.ravel()], 1))
        t, _ = self.solid.trace(O, D)
        hit = ~np.isnan(t)
        P = O + np.where(hit, t, 1e4)[:, None] * D
        N = np.zeros_like(P)
        if hit.any():
            N[hit] = self.solid.normal(P[hit])
        return P.reshape(shape + (3,)), N.reshape(shape + (3,))

    def outline(self, W, H, pad=4):
        """The solid's silhouette on a W x H canvas, and for each spot inside it which part
        (group) of the solid shows there (-1 outside)."""
        s = self.solid
        c = s.world(s.centre)
        cxy = self.cam.project(c)
        rpx = s.radius * s.k * self.cam.px_per_cm(c) * 1.08 + pad
        x0, x1 = int(max(0, np.floor(cxy[0] - rpx))), int(min(W, np.ceil(cxy[0] + rpx)))
        y0, y1 = int(max(0, np.floor(cxy[1] - rpx))), int(min(H, np.ceil(cxy[1] + rpx)))
        group = np.full((H, W), -1, dtype=np.int32)
        if x1 <= x0 or y1 <= y0:
            return group
        X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        O, D = self.rays(np.stack([X.ravel(), Y.ravel()], 1))
        t, _ = s.trace(O, D)
        hit = ~np.isnan(t)
        g = np.full(len(t), -1, dtype=np.int32)
        if hit.any():
            g[hit] = s.group(O[hit] + t[hit, None] * D[hit])
        group[y0:y1, x0:x1] = g.reshape(Y.shape)
        return group
