"""The light logic of the picture: one window, the dim room, and what everything throws on its neighbours.

This is not a renderer; no image is made here. It answers the questions a painter asks before
mixing the colour for one stroke: how much of the window does this spot see, how shut in is it
by the things round it, what colour comes back onto it from its neighbours, and (for anything
polished) what would I see reflected in it from where I stand?
"""
import numpy as np

from .canvas import linear_to_srgb, srgb_to_linear, as_rgb
from .geom import normalize


def lin(c):
    """sRGB colour ('#rrggbb' or triple) -> linear reflectance."""
    return srgb_to_linear(as_rgb(c))


def luma(rgb):
    """Rough lightness of sRGB colours (..., 3)."""
    rgb = np.asarray(rgb)
    return 0.30 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]


_FAN = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                 [0.6, 0.6, 0.6], [-0.6, -0.6, 0.6], [0.6, -0.6, -0.6], [-0.6, 0.6, -0.6]], dtype=np.float64) * 0.9


class Lighting:
    def __init__(self, light, cam, key=(1.00, 0.95, 0.84),
                 sky=(0.080, 0.086, 0.100),     # cool spill from the window side, from above
                 floor=(0.30, 0.185, 0.095),    # warm light thrown up by the table
                 room=(0.026, 0.022, 0.019),    # the dark of the room
                 exposure=1.0, wrap=0.16,
                 window=(0.20, 0.27)):          # half-width and half-height of the window (as tangents)
        self.light = light
        self.cam = cam
        self.key = np.asarray(key)
        self.sky = np.asarray(sky)
        self.floor = np.asarray(floor)
        self.room = np.asarray(room)
        self.exposure = exposure
        self.wrap = wrap
        self.window = window
        self.pool = None          # f(P) -> 0..1: where the window's light actually falls
        self.glows = []           # coloured neighbours (geom.Glow)
        self.table_look = None    # f(Q) -> linear colour of the table top as seen at points Q
        self.wall_look = None     # f(Q) -> linear colour of the wall at points Q
        self.wall_z = None
        up = np.array([0.0, 1.0, 0.0])
        self._wx = normalize(np.cross(up, light.L))
        self._wy = np.cross(light.L, self._wx)

    # ---- working near one form: only the solids and neighbours that matter there
    def near(self, center, radius):
        other = Lighting.__new__(Lighting)
        other.__dict__.update(self.__dict__)
        other.light = self.light.near(center, radius)
        C = np.asarray(center, dtype=np.float64)
        other.glows = [g for g in self.glows if np.linalg.norm(g.c - C) < radius + g.r + 26.0]
        return other

    # ---- the pieces
    def ambient(self, N, P=None):
        """The light that is not the window's direct beam: the cool spill from the window side,
        the dim room, and the warm light the lit table throws back up at anything standing on it."""
        up = np.clip(N[:, 1], 0, 1)[:, None]
        toward_window = np.clip(N @ self.light.L, -1, 1)[:, None] * 0.5 + 0.5
        sees_table = (0.5 - 0.5 * N[:, 1])[:, None] ** 1.3          # how much of what it faces is table
        table_lit = 1.0
        if P is not None and self.pool is not None:
            Q = np.array(P, dtype=np.float64)
            low = np.clip(1 - Q[:, 1] / 14.0, 0.25, 1)              # less of it the higher up
            table_lit = (self.pool(Q) * low)[:, None]
        return self.room + self.sky * (0.35 * up + 0.65 * toward_window ** 2) + self.floor * sees_table * table_lit

    def bounce(self, P, N, skip=()):
        """Colour thrown onto a surface by its lit neighbours."""
        out = np.zeros((len(P), 3))
        L = self.light.L
        for g in self.glows:
            if any(g.name.startswith(s) or s.startswith(g.name) for s in skip):     # nothing lights itself
                continue
            v = g.c - P
            d2 = np.sum(v * v, axis=1)
            d = np.sqrt(np.maximum(d2, 1e-9))
            facing = np.clip(np.sum(v * N, axis=1) / d, 0, 1)
            # the side of the neighbour that faces this spot is lit if it also faces the window
            side = 0.18 + 0.82 * np.clip(-(v @ L) / d, 0, 1)
            w = g.r ** 2 / (d2 + g.r ** 2) * facing * side
            out += g.colour * w[:, None]
        return out * self.key * 0.9

    def tone(self, linear):
        # gentle shoulder: near-linear in the darks and middles, lights roll off instead of clipping
        x = self.exposure * np.maximum(linear, 0)
        return linear_to_srgb(x / (1 + x ** 2.2) ** (1 / 2.2))

    def shade(self, albedo, P, N, skip=(), spec=0.0, gloss=60.0, spec_tint=(1, 1, 1), ao=1.0,
              wrap=None, floor_gain=1.0, lit=None, emit=None, raw=False, crowd=1.0):
        """Colour (sRGB) of matte or satin surface points P (N, 3) with normals N and linear albedo.

        ao: extra hand-set shutting-in (0..1); crowd: how strongly nearby solids darken it.
        """
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        N = normalize(np.asarray(N, dtype=np.float64).reshape(-1, 3))
        albedo = np.broadcast_to(np.asarray(albedo, dtype=np.float64), P.shape)
        L = self.light.L
        wrap = self.wrap if wrap is None else wrap
        ndl = N @ L
        vis = self.light.lit(P, skip) if lit is None else lit
        pool = self.pool(P) if self.pool is not None else 1.0
        diff = np.clip((ndl + wrap) / (1 + wrap), 0, 1) * vis * pool
        open_ = self.light.open_to_room(P, N, skip, reach=crowd) if crowd > 0 else 1.0
        amb = self.ambient(N, P)
        amb = amb + self.floor * (floor_gain - 1) * np.clip(-N[:, 1], 0, 1)[:, None]
        shut = np.asarray(ao).reshape(-1, 1) * np.asarray(open_).reshape(-1, 1)
        amb = self.room * 0.5 + (amb - self.room * 0.5) * shut
        out = albedo * (self.key * diff[:, None] + amb + self.bounce(P, N, skip))
        if spec > 0:
            V = self.cam.to_eye(P)
            Hv = normalize(V + L)
            s = np.clip(np.sum(N * Hv, axis=1), 0, 1) ** gloss * (ndl > 0) * vis * pool
            out = out + spec * s[:, None] * self.key * np.asarray(spec_tint)
        if emit is not None:
            out = out + emit
        return out if raw else self.tone(out)

    # ---- what a polished thing shows: the room, seen along the mirrored line of sight
    def surroundings(self, P, R, skip=()):
        """Linear colour seen from points P looking along unit directions R: the window, the wall,
        the table, a coloured neighbour, or the dark of the room."""
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        R = normalize(np.asarray(R, dtype=np.float64).reshape(-1, 3))
        n = len(P)
        L = self.light.L
        out = np.tile(self.room * 1.2, (n, 1))
        dist = np.full(n, 1e9)
        # the window: a bright rectangle with a cross of glazing bars, high and to the left
        rl = R @ L
        a = (R @ self._wx) / np.maximum(rl, 1e-6)
        b = (R @ self._wy) / np.maximum(rl, 1e-6)
        hw, hh = self.window
        inside = (rl > 0) & (np.abs(a) < hw) & (np.abs(b) < hh)
        bars = (np.abs(a) < hw * 0.07) | (np.abs(b - hh * 0.1) < hh * 0.05)
        glare = np.exp(-((a / (hw * 2.2)) ** 2 + (b / (hh * 2.2)) ** 2)) * (rl > 0)      # light spilling round it
        out = out + (self.key * 0.22)[None, :] * glare[:, None]
        out[inside & ~bars] = self.key * 7.0
        out[inside & bars] = self.key * 1.2
        # the table top
        if self.table_look is not None:
            t = np.where(R[:, 1] < -1e-4, -P[:, 1] / np.minimum(R[:, 1], -1e-4), 1e9)
            hit = (t > 1e-3) & (t < dist) & ~inside
            if hit.any():
                Q = P[hit] + t[hit, None] * R[hit]
                out[hit] = self.table_look(Q)
                dist[hit] = t[hit]
        # the wall behind
        if self.wall_look is not None and self.wall_z is not None:
            t = np.where(R[:, 2] < -1e-4, (self.wall_z - P[:, 2]) / np.minimum(R[:, 2], -1e-4), 1e9)
            hit = (t > 1e-3) & (t < dist) & ~inside
            if hit.any():
                Q = P[hit] + t[hit, None] * R[hit]
                ok = Q[:, 1] > 0
                idx = np.nonzero(hit)[0][ok]
                out[idx] = self.wall_look(Q[ok])
                dist[idx] = t[hit][ok]
        # coloured neighbours, each standing in as a ball
        for g in self.glows:
            if any(g.name.startswith(s) or s.startswith(g.name) for s in skip):
                continue
            oc = P - g.c
            bq = np.sum(oc * R, axis=1)
            disc = bq * bq - (np.sum(oc * oc, axis=1) - g.r ** 2)
            t = -bq - np.sqrt(np.maximum(disc, 0))
            hit = (disc > 0) & (t > 1e-3) & (t < dist)
            if hit.any():
                nrm = (oc[hit] + t[hit, None] * R[hit]) / g.r
                lit_side = 0.10 + np.clip(nrm @ L, 0, 1)
                pool = self.pool(g.c[None, :])[0] if self.pool is not None else 1.0
                out[hit] = g.colour * self.key * (lit_side * pool)[:, None]
                dist[hit] = t[hit]
        return out

    def mirror(self, tint, P, N, skip=(), rough=0.0, dirt=None, rng=None, taps=1, raw=False, lit=None):
        """Colour of bare metal or any polished surface: its own tint times what it reflects.

        rough: how far the reflection is smeared (0 = a true mirror); taps: how many nearby
        directions are averaged when it is smeared; dirt: optional (colour, amount per point) of
        dull tarnish or grime, which is lit like an ordinary matte surface.
        """
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        N = normalize(np.asarray(N, dtype=np.float64).reshape(-1, 3))
        tint = np.broadcast_to(np.asarray(tint, dtype=np.float64), P.shape)
        V = self.cam.to_eye(P)
        acc = np.zeros((len(P), 3))
        # a smeared reflection is the average of what a small fan of mirrors would show
        fan = _FAN if (rough > 0 and taps > 1) else _FAN[:1]
        for off in fan:
            Nj = N if rough <= 0 else normalize(N + rough * off)
            R = 2 * np.sum(Nj * V, axis=1, keepdims=True) * Nj - V
            acc += self.surroundings(P, R, skip)
        out = tint * acc / len(fan)
        # the window itself only shows where the window's light really arrives
        vis = self.light.lit(P, skip) if lit is None else lit
        bright = np.max(out, axis=1) > 1.5
        out[bright] = out[bright] * (0.05 + 0.95 * vis[bright, None])
        if dirt is not None:
            d, amount = dirt
            pool = self.pool(P) if self.pool is not None else 1.0
            ndl = np.clip(N @ self.light.L, 0, 1) * vis * pool
            out = out * (1 - amount[:, None]) + amount[:, None] * d * (self.key * ndl[:, None] + self.ambient(N, P))
        return out if raw else self.tone(out)

    # kept for older code: a broad-lobed metal
    def metal(self, tint, P, N, skip=(), gloss=9.0, shine=1.0, env=1.0, dirt=None, lit=None, raw=False):
        rough = float(np.clip(1.6 / np.sqrt(max(gloss, 1.0)), 0.05, 0.6))
        return self.mirror(tint, P, N, skip=skip, rough=rough, taps=5, dirt=dirt, raw=raw, lit=lit)
