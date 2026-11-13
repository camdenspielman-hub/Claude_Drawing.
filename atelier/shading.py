"""The light logic of the picture: one window, the dim room, and the warm table.

This is not a renderer. It answers the question a painter asks before mixing a
colour for a stroke: "this bit of surface faces that way, in this light - how
light, how warm?" Every stroke asks once, for the spot it is about to cover.
"""
import numpy as np

from .canvas import linear_to_srgb, srgb_to_linear, as_rgb
from .geom import normalize


def lin(c):
    """sRGB colour ('#rrggbb' or triple) -> linear reflectance."""
    return srgb_to_linear(as_rgb(c))


class Lighting:
    def __init__(self, light, cam, key=(0.88, 0.84, 0.74),
                 sky=(0.20, 0.21, 0.23),        # cool spill from the window side, from above
                 floor=(0.30, 0.20, 0.12),      # warm light thrown up by the table
                 room=(0.07, 0.06, 0.055),      # the dark of the room
                 exposure=1.0):
        self.light = light
        self.cam = cam
        self.key = np.asarray(key)
        self.sky = np.asarray(sky)
        self.floor = np.asarray(floor)
        self.room = np.asarray(room)
        self.exposure = exposure
        self.pool = None      # optional f(P) -> 0..1: where the window's light actually falls

    def ambient(self, N):
        up = np.clip(N[:, 1], 0, 1)[:, None]
        dn = np.clip(-N[:, 1], 0, 1)[:, None]
        toward_window = np.clip(N @ self.light.L, -1, 1)[:, None] * 0.5 + 0.5
        return self.room + self.sky * (0.35 * up + 0.65 * toward_window ** 2) + self.floor * dn

    def tone(self, linear):
        # gentle shoulder: near-linear in the darks and middles, lights roll off instead of clipping
        x = self.exposure * np.maximum(linear, 0)
        return linear_to_srgb(x / (1 + x ** 2.2) ** (1 / 2.2))

    def shade(self, albedo, P, N, skip=(), spec=0.0, gloss=60.0, spec_tint=(1, 1, 1), ao=1.0,
              wrap=0.0, floor_gain=1.0, lit=None, emit=None, raw=False):
        """Colour (sRGB) of surface points P (N, 3) with normals N and linear albedo."""
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        N = normalize(np.asarray(N, dtype=np.float64).reshape(-1, 3))
        albedo = np.broadcast_to(np.asarray(albedo, dtype=np.float64), P.shape)
        L = self.light.L
        ndl = N @ L
        vis = self.light.lit(P, skip) if lit is None else lit
        diff = np.clip((ndl + wrap) / (1 + wrap), 0, 1) * vis
        if self.pool is not None:
            diff = diff * self.pool(P)
        amb = self.ambient(N)
        amb = self.room + (amb - self.room) * np.asarray(ao).reshape(-1, 1) if np.ndim(ao) else amb * ao
        amb = amb + self.floor * (floor_gain - 1) * np.clip(-N[:, 1], 0, 1)[:, None]
        out = albedo * (self.key * diff[:, None] + amb)
        if spec > 0:
            V = self.cam.to_eye(P)
            Hv = normalize(V + L)
            s = np.clip(np.sum(N * Hv, axis=1), 0, 1) ** gloss * (ndl > 0) * vis
            if self.pool is not None:
                s = s * self.pool(P)
            out = out + spec * s[:, None] * self.key * np.asarray(spec_tint)
        if emit is not None:
            out = out + emit
        return out if raw else self.tone(out)

    def metal(self, tint, P, N, skip=(), gloss=9.0, shine=1.0, env=1.0, dirt=None, lit=None, raw=False):
        """Colour of bare metal. Metal has almost no colour of its own: it shows the window where
        it mirrors the window, the dark room elsewhere, and the warm table from below.
        tint: the metal's colour (linear); dirt: optional (N, 3) dull colour mixed in (tarnish, grime)."""
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        N = normalize(np.asarray(N, dtype=np.float64).reshape(-1, 3))
        tint = np.broadcast_to(np.asarray(tint, dtype=np.float64), P.shape)
        V = self.cam.to_eye(P)
        R = 2 * np.sum(N * V, axis=1, keepdims=True) * N - V
        vis = self.light.lit(P, skip) if lit is None else lit
        if self.pool is not None:
            vis = vis * self.pool(P)
        window = np.clip(R @ self.light.L, 0, 1) ** gloss * vis * (N @ self.light.L > 0)
        out = tint * (shine * 1.6 * window[:, None] * self.key + env * self.ambient(R))
        if dirt is not None:
            d, amount = dirt
            ndl = np.clip(N @ self.light.L, 0, 1) * vis
            out = out * (1 - amount[:, None]) + amount[:, None] * d * (self.key * ndl[:, None] + self.ambient(N))
        return out if raw else self.tone(out)

