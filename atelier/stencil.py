"""Stencils: the drawn outline of a thing, as the brush needs to know it.

A stencil never paints anything. It records which part of the canvas an object
occupies, so that strokes can stop short of its edge (by the width of the
brush) and so that things behind it can be left unpainted where it will sit -
the 'reserve' an old master left in the background for each object.
"""
import numpy as np
from PIL import Image, ImageDraw

from .canvas import distance_inside


class Stencil:
    def __init__(self, mask):
        self.mask = np.ascontiguousarray(mask, dtype=bool)
        self.H, self.W = self.mask.shape
        self._dist = None

    @staticmethod
    def empty(W, H):
        return Stencil(np.zeros((H, W), bool))

    @staticmethod
    def full(W, H):
        return Stencil(np.ones((H, W), bool))

    @staticmethod
    def from_polys(polys, W, H):
        im = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(im)
        for p in polys:
            p = np.asarray(p, dtype=np.float64)
            if len(p) >= 3:
                dr.polygon([(float(x), float(y)) for x, y in p], fill=255)
        return Stencil(np.asarray(im) > 0)

    @staticmethod
    def from_surface(cam, surf, urange, vrange, W, H, grid=96, visible=None, min_facing=0.0):
        """The canvas area covered by the visible, front-facing part of a parametric surface."""
        gu = np.linspace(urange[0], urange[1], grid + 1)
        gv = np.linspace(vrange[0], vrange[1], grid + 1)
        U, V = np.meshgrid(gu, gv, indexing="ij")
        P, N = surf(U, V)
        xy = cam.project(P)
        face = np.sum(N * cam.to_eye(P), axis=-1) > min_facing
        if visible is not None:
            face &= visible(P, N, U, V)
        # a cell counts if any of its corners is visible
        ok = face[:-1, :-1] | face[1:, :-1] | face[:-1, 1:] | face[1:, 1:]
        im = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(im)
        ii, jj = np.nonzero(ok)
        for i, j in zip(ii, jj):
            q = [xy[i, j], xy[i + 1, j], xy[i + 1, j + 1], xy[i, j + 1]]
            dr.polygon([(float(x), float(y)) for x, y in q], fill=255)
        return Stencil(np.asarray(im) > 0)

    # --- combining
    def __or__(self, o):
        return Stencil(self.mask | o.mask)

    def __and__(self, o):
        return Stencil(self.mask & o.mask)

    def __sub__(self, o):
        return Stencil(self.mask & ~o.mask)

    def __invert__(self):
        return Stencil(~self.mask)

    # --- queries
    @property
    def dist(self):
        if self._dist is None:
            self._dist = distance_inside(self.mask)
        return self._dist

    def inside(self, xy, margin=0.0):
        """True where canvas points lie inside the outline by more than `margin` px."""
        xy = np.asarray(xy)
        x = np.round(xy[..., 0] - 0.5).astype(np.int64)
        y = np.round(xy[..., 1] - 0.5).astype(np.int64)
        ok = (x >= 0) & (x < self.W) & (y >= 0) & (y < self.H)
        xc, yc = np.clip(x, 0, self.W - 1), np.clip(y, 0, self.H - 1)
        if margin <= 0:
            return ok & self.mask[yc, xc]
        return ok & (self.dist[yc, xc] > margin)

    def depth(self, xy):
        """Distance in px from canvas points to the outline (0 outside)."""
        xy = np.asarray(xy)
        x = np.clip(np.round(xy[..., 0] - 0.5).astype(np.int64), 0, self.W - 1)
        y = np.clip(np.round(xy[..., 1] - 0.5).astype(np.int64), 0, self.H - 1)
        return self.dist[y, x]

    def bbox(self, pad=0):
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        return (max(0, xs.min() - pad), max(0, ys.min() - pad),
                min(self.W, xs.max() + 1 + pad), min(self.H, ys.max() + 1 + pad))

    def any(self):
        return bool(self.mask.any())
