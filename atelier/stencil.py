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
    def _crop(self):
        """Distance from the outline for each inside pixel, kept only over the shape's bounding box."""
        if self._dist is None:
            ys, xs = np.nonzero(self.mask)
            if len(xs) == 0:
                self._dist = np.zeros((1, 1), np.float32)
                self._off = (0, 0)
            else:
                x0, y0 = max(0, xs.min() - 1), max(0, ys.min() - 1)
                x1, y1 = min(self.W, xs.max() + 2), min(self.H, ys.max() + 2)
                self._dist = distance_inside(self.mask[y0:y1, x0:x1])
                self._off = (x0, y0)
        return self._dist, self._off

    @property
    def dist(self):
        """Full-canvas array of distances (costly in memory; prefer depth())."""
        d, (x0, y0) = self._crop()
        out = np.zeros((self.H, self.W), np.float32)
        out[y0:y0 + d.shape[0], x0:x0 + d.shape[1]] = d
        return out

    def _pix(self, xy):
        xy = np.asarray(xy)
        return (np.round(xy[..., 0] - 0.5).astype(np.int64), np.round(xy[..., 1] - 0.5).astype(np.int64))

    def _depth_at(self, x, y):
        d, (x0, y0) = self._crop()
        xi, yi = x - x0, y - y0
        ok = (xi >= 0) & (xi < d.shape[1]) & (yi >= 0) & (yi < d.shape[0])
        out = np.zeros(np.shape(x), np.float32)
        out[ok] = d[yi[ok], xi[ok]]
        return out

    def inside(self, xy, margin=0.0):
        """True where canvas points lie inside the outline by more than `margin` px."""
        x, y = self._pix(xy)
        ok = (x >= 0) & (x < self.W) & (y >= 0) & (y < self.H)
        if margin <= 0:
            return ok & self.mask[np.clip(y, 0, self.H - 1), np.clip(x, 0, self.W - 1)]
        return ok & (self._depth_at(x, y) > margin)

    def depth(self, xy):
        """Distance in px from canvas points to the outline (0 outside)."""
        x, y = self._pix(xy)
        return self._depth_at(x, y)

    def inward(self, xy):
        """Unit vectors pointing from canvas points into the shape (straight away from its nearest edge)."""
        x, y = self._pix(xy)
        g = np.stack([self._depth_at(x + 1, y) - self._depth_at(x - 1, y),
                      self._depth_at(x, y + 1) - self._depth_at(x, y - 1)], axis=-1).astype(np.float64)
        n = np.linalg.norm(g, axis=-1, keepdims=True)
        return g / np.maximum(n, 1e-6)

    def edge_points(self, spacing, rng, band=1.6):
        """Points scattered along the outline, about `spacing` px apart, just inside it."""
        d, (x0, y0) = self._crop()
        ys, xs = np.nonzero((d > 0) & (d <= band))
        if len(xs) == 0:
            return np.zeros((0, 2))
        keep = rng.random(len(xs)) < min(1.0, 1.0 / max(spacing, 1.0))
        return np.stack([xs[keep] + x0 + 0.5, ys[keep] + y0 + 0.5], axis=1)

    def area(self):
        return int(self.mask.sum())

    def bbox(self, pad=0):
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        return (max(0, xs.min() - pad), max(0, ys.min() - pad),
                min(self.W, xs.max() + 1 + pad), min(self.H, ys.max() + 1 + pad))

    def any(self):
        return bool(self.mask.any())
