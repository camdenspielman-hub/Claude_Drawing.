"""The setting: a plastered wall and an old oak table.

Painted the way Claesz and Heda painted theirs. The wall is a thin, uneven, low glow, lightest
on the side away from the window and darker in a band just above the table. The table top is
a plain warm plane, paler where hands have rubbed it near the front, with one joint between
boards, a crack, and old knife marks; its front face is a darker band under a worn pale edge.
"""
import numpy as np

from atelier import forms
from atelier.brushwork import Batch
from atelier.canvas import Brush, Paint
from atelier.forms import fbm3, noise3
from atelier.geom import smoothstep
from atelier.shading import lin
from atelier.studio import Form
from . import scene as L

OAK = lin("#7c5d3c")
OAK_DARK = lin("#3f2a18")
WALL_HUE = np.array([1.10, 0.94, 0.60])     # olive umber
JOINT_Z = 3.3                               # where two boards of the top meet


def window_pool(P):
    """Where the window's light falls: fullest on the front of the table a little left of centre,
    dying away to the sides and into the depth of the room."""
    x, z = P[:, 0], P[:, 2]
    side = np.exp(-((x - 1.5) / 27.0) ** 2)
    depth = smoothstep(-26.0, 9.0, z) ** 1.5
    return 0.06 + 0.94 * side * (0.10 + 0.90 * depth)


def wall_linear(P, lit=1.0):
    """The wall's tone (linear) at points on it."""
    x, y = P[:, 0], P[:, 1]
    glow = np.exp(-(((x - 11.0) / 25.0) ** 2 + ((y - 12.0) / 15.0) ** 2))
    v = 0.0055 + 0.040 * glow
    q = np.stack([x * 0.16, y * 0.16, 0 * x], 1)
    patches = 0.87 + 0.26 * fbm3(q * 0.6, 1.0, 2, 41)                 # dry, uneven scumbling
    band = 1 - 0.38 * np.exp(-(y / 2.4) ** 2)                         # darker just above the table
    warm = 0.5 + 0.5 * fbm3(q * 0.5 + 7.0, 1.0, 2, 43)
    hue = WALL_HUE * (1 + 0.10 * (warm[:, None] - 0.5) * np.array([1.0, 0.2, -1.0]))
    return (v * patches * band * (0.55 + 0.45 * np.asarray(lit)))[:, None] * hue


def oak_albedo(P, front=False):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    across = y if front else z
    # the grain is quiet: long faint streaks, slow drifts of tone from board to board
    streak = fbm3(np.stack([x * 0.20, across * 5.0, 0 * x + (3.0 if front else 0.0)], 1), 1.0, 3, 3)
    drift = noise3(np.stack([x * 0.045, across * 0.5, 0 * x], 1), 1.0, 7)
    a = OAK * (0.86 + 0.24 * streak)[:, None] * (0.90 + 0.20 * drift)[:, None]
    stain = smoothstep(0.60, 0.85, fbm3(np.stack([x * 0.22, across * 0.3, 0 * x + 5.0], 1), 1.0, 3, 13))[:, None]
    a = a * (1 - 0.30 * stain) + OAK_DARK * 0.30 * stain
    if front:
        return a * 0.34
    # rubbed pale and matte toward the front edge; grimy toward the back
    a = a * (1 + 0.16 * smoothstep(L.TABLE_FRONT - 14.0, L.TABLE_FRONT, z))[:, None]
    a = a * ((1 - 0.24 * smoothstep(-6.0, -22.0, z))[:, None] * (1 + smoothstep(-8.0, -22.0, z)[:, None] * np.array([0.10, 0.0, -0.16])))
    # the joint between two boards: a wavering dark line, and the boards either side a shade apart
    joint = JOINT_Z + 0.10 * np.sin(x * 0.19) + 0.05 * np.sin(x * 0.71)
    a = a * (1 - 0.80 * np.exp(-((z - joint) / 0.055) ** 2))[:, None]                 # the gap
    a = a * (1 + 0.22 * np.exp(-((z - joint - 0.13) / 0.07) ** 2))[:, None]           # the near board's rubbed edge
    a = a * (1 + 0.05 * np.sign(z - joint))[:, None]
    return a


class Setting:
    softness = {"wall": (0.55, 0.2), "table": (0.50, 0.0), "front": (0.25, 0.0)}

    def __init__(self, sc):
        self.sc = sc
        lt = sc.lt
        lt.pool = window_pool
        lt.table_look = lambda Q: OAK * 0.9 * (lt.key * lt.light.L[1] * window_pool(Q)[:, None] + 0.07)
        lt.wall_look = lambda Q: wall_linear(Q)
        wall = forms.plane([0, 0, L.WALL_Z], [1.0, 0, 0], [0, 1.0, 0])
        top = forms.plane([0, 0, 0], [1.0, 0, 0], [0, 0, 1.0], flip=True)
        front = forms.plane([0, 0, L.TABLE_FRONT], [1.0, 0, 0], [0, -1.0, 0], flip=True)

        def wall_col(P, N, u, v):
            return lt.tone(wall_linear(P, lt.light.lit(P)))

        def top_col(P, N, u, v):
            return lt.shade(oak_albedo(P), P, N, spec=0.03, gloss=6, crowd=1.0)

        def front_col(P, N, u, v):
            edge = smoothstep(0.35, 0.0, v)[:, None]              # the worn arris catches the light
            return lt.shade(oak_albedo(P, front=True) * (1 + 1.6 * edge), P, N, crowd=0.6)
        self.forms = [
            Form("wall", wall, (-70, 70), (0, 52), wall_col, size=30, grid=36, finish=0.2, along="u", broken=0.0,
                 coverage=1.7, thick=(1.1, 1.25), tol=0.02,
                 flat=(np.array([0, 0, L.WALL_Z]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))),
            Form("table", top, (-52, 52), (L.TABLE_BACK, L.TABLE_FRONT), top_col, size=15, grid=80, finish=1.0, along="u",
                 broken=0.05, coverage=1.3, tol=0.02, thick=(0.45, 1.2),
                 flat=(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))),
            Form("front", front, (-52, 52), (0, L.TABLE_DROP), front_col, size=9, grid=40, finish=0.6, along="u",
                 broken=0.05, coverage=1.3,
                 flat=(np.array([0, 0, L.TABLE_FRONT]), np.array([1.0, 0, 0]), np.array([0, -1.0, 0]))),
        ]

    def occluders(self):
        return []

    def glows(self):
        return []

    def groups(self):
        return [(1e9 - i, [f]) for i, f in enumerate(self.forms)]

    def register(self, studio):
        studio.softness_of.update(self.softness)
        studio.hook("dead", lambda st: self.blend_wall(st, edges=False))
        studio.hook("colour", self.blend_wall)
        studio.hook("colour", self.lose_far_edge)
        studio.hook("model", self.lose_far_edge)
        studio.hook("model", self.blend_table)
        studio.hook("finish", self.wear)

    def lose_far_edge(self, studio):
        """Where the table meets the wall, far back, the line is let go: firm for a stretch, then lost."""
        sc = self.sc
        cv, cam, rng, px = sc.cv, sc.cam, sc.rng, sc.px
        table, wall = studio.regions["table"], studio.regions["wall"]
        xs = np.linspace(-52, 52, 260)
        line = cam.project(np.stack([xs, 0 * xs, np.full_like(xs, L.TABLE_BACK)], 1))
        lose = 0.60 + 0.40 * np.clip(0.5 + 1.3 * (fbm3(np.stack([xs * 0.09, 0 * xs, 0 * xs + 4.0], 1), 1.0, 2, 57) - 0.5) * 2, 0, 1)
        lose = np.maximum(lose, smoothstep(14.0, 34.0, np.abs(xs - 2.0)))          # always lost toward the sides
        batch = Batch(cv, Brush(kind="blender", radius=px(1.0), hardness=0.05, pickup=0.5, load=0.5))
        for i in range(0, len(xs) - 6, 3):
            seg = line[i:i + 7]
            mid = seg[3]
            if not (table.inside((mid + [0, px(6)])[None])[0] and wall.inside((mid - [0, px(6)])[None])[0]):
                continue                                    # something stands in front of the edge here
            if lose[i + 3] < 0.3:
                continue
            for dy in (-0.5, 0.0, 0.6):
                batch.add(seg + [0, px(dy * 5 * lose[i + 3])], np.zeros(3), 1.0, 2.5 + 4.5 * lose[i + 3])
        batch.flush(rng=rng)

    def blend_table(self, studio):
        """Soften the table top with a clean brush drawn along the grain, so cast shadows have soft
        edges instead of crumbled ones (their sharp contact accents are put back at the finish)."""
        from atelier.brushwork import fill_field
        sc = self.sc
        px = sc.px
        table = studio.regions["table"]
        for r in (7.0, 4.5):
            fill_field(sc.cv, table.bbox(), None, lambda xy: 0.12 * np.sin(xy[:, 1] / px(60.0)),
                       Brush(kind="blender", radius=px(r), hardness=0.0, pickup=0.32, load=0.5), sc.rng, length=px(r * 6),
                       spacing=px(r * 0.6), stencil=table, margin=px(r * 1.1), paint=Paint.opaque("#000000"), curl=0.08)

    def blend_wall(self, studio, edges=True):
        """Work the wall together with a big soft brush while it is wet, so it is one thin uneven glow;
        then carry that same paint up to everything standing against it, so no collar is left round anything."""
        from atelier.brushwork import fill_field
        sc = self.sc
        cv, rng, px = sc.cv, sc.rng, sc.px
        wall = studio.regions["wall"]
        wf = studio.form("wall")
        if edges:
            wf.cut_in = False                      # the wall's edges are done here, while it is wet
            studio._cut_in(wf)

        def swirl(xy, k=0):
            # the brush wanders every way over the wall, as when a ground is scumbled in
            q = np.stack([xy[:, 0] / px(420.0), xy[:, 1] / px(420.0), np.full(len(xy), 3.0 + k)], 1)
            return 2.6 * np.pi * fbm3(q, 1.0, 2, 71 + k)
        for i, r in enumerate((36.0, 26.0, 18.0, 12.0)):
            fill_field(cv, wall.bbox(), None, lambda xy, i=i: swirl(xy, i),
                       Brush(kind="blender", radius=px(r), hardness=0.0, pickup=0.55, load=0.5), rng, length=px(r * 5),
                       spacing=px(r * 0.45), stencil=wall, margin=px(r * 0.8), paint=Paint.opaque("#000000"), curl=0.25)
        if not edges:
            return
        # match the wall beside each object to the wall a little further out: look there, mix that, lay it in
        for r, band, reach in ((9.0, 40.0, 54.0), (5.0, 18.0, 34.0), (2.5, 8.0, 20.0)):
            studio._look = None
            pts = wall.edge_points(px(r * 0.7), rng, band=px(band))
            if len(pts) == 0:
                continue
            n = wall.inward(pts)
            src = pts + n * np.maximum(px(reach) - wall.depth(pts), 0.0)[:, None]
            ok = wall.inside(src, px(reach * 0.55))
            pts, n, cols = pts[ok], n[ok], studio.seen(src[ok])
            t = np.stack([-n[:, 1], n[:, 0]], axis=1)
            batch = Batch(cv, Brush(radius=px(r), hardness=0.3, thick=1.25, pickup=0.35), wf.body)
            ln = px(r * 3.2)
            for i in range(len(pts)):
                batch.add(np.array([pts[i] - t[i] * ln / 2, pts[i] + t[i] * ln / 2]), cols[i], [0.7, 1.0])
            batch.flush(rng=rng)
        studio._look = None
        # and close in to everything that stands against it with smaller and smaller clean brushes
        for i, r in enumerate((8.0, 5.0, 3.0)):
            near = lambda xy, r=r: wall.depth(xy) < px(r * 7.0)
            fill_field(cv, wall.bbox(), None, lambda xy, i=i: swirl(xy, i + 5),
                       Brush(kind="blender", radius=px(r), hardness=0.0, pickup=0.5, load=0.5), rng, length=px(r * 5),
                       spacing=px(r * 0.5), stencil=wall, margin=px(r * 0.75), inside=near, paint=Paint.opaque("#000000"), curl=0.3)
        studio._to_border(wf, 10.0)

    def wear(self, studio):
        """What use has done to the table: knife marks, a crack running in from the edge, the pale
        rubbed arris with its nicks, a ring where something wet once stood."""
        sc = self.sc
        cv, cam, lt, rng, px = sc.cv, sc.cam, sc.lt, sc.rng, sc.px
        table = studio.regions["table"]

        def on_table(pts):
            xy = cam.project(pts)
            return xy, table.inside(xy, px(1.0))
        tone = lambda P: lt.shade(oak_albedo(P), P, np.tile([0.0, 1.0, 0.0], (len(P), 1)), crowd=0.0)
        # knife marks: short straight scores at all angles, each a dark cut with a pale lip
        dark = Batch(cv, Brush(radius=px(0.6), hardness=0.5, thick=0.5, pickup=0.2, press_radius=0.6))
        pale = Batch(cv, Brush(radius=px(0.5), hardness=0.5, thick=0.35, pickup=0.2, press_radius=0.6))
        for _ in range(12):
            c = np.array([rng.uniform(-20, 22), 0.0, rng.uniform(-6, 16.4)])
            ang, ln = rng.uniform(0, np.pi), rng.uniform(0.4, 1.8)
            d = np.array([np.cos(ang), 0, np.sin(ang)])
            pts = c + np.linspace(-ln / 2, ln / 2, 9)[:, None] * d
            xy, ok = on_table(pts)
            if not ok.all():
                continue
            col = tone(c[None])[0]
            dark.add(xy, np.clip(col * 0.84, 0, 1), np.sin(np.linspace(0.3, np.pi - 0.3, 9)), tscale=rng.uniform(0.3, 0.7))
            pale.add(xy + [0, px(1.0)], np.clip(col * 1.08, 0, 1), np.sin(np.linspace(0.3, np.pi - 0.3, 9)) * 0.6,
                     tscale=rng.uniform(0.3, 0.7))
        dark.flush()
        pale.flush()
        cv.dry_now()
        # the arris: rubbed pale by years of sleeves, broken by nicks
        xs = np.linspace(-52, 52, 620)
        edge3 = np.stack([xs, np.zeros_like(xs), np.full_like(xs, L.TABLE_FRONT)], 1)
        edge = cam.project(edge3)
        free = studio.regions["front"].inside(edge + [0, px(2.5)]) | table.inside(edge - [0, px(2.5)])
        wear = np.clip(0.55 + 1.1 * (fbm3(np.stack([xs * 0.4, 0 * xs, 0 * xs + 2.0], 1), 1.0, 3, 51) - 0.5), 0, 1)
        col = lt.shade(OAK * 1.45, edge3, np.tile([0.0, 0.75, 0.66], (len(xs), 1)), crowd=0.8)
        for a0 in range(0, len(xs) - 10, 9):
            seg = slice(a0, a0 + 11)
            if free[seg].all() and rng.random() > 0.08:
                cv.stroke(edge[seg] + [0, px(0.4)], Paint.opaque(np.clip(col[a0 + 5], 0, 1)),
                          Brush(kind="bristle", radius=px(1.8), nbristle=9, bristle_r=px(0.75), jitter=0.2, thick=0.9,
                                pickup=0.3, tooth=0.6, load=6), wear[seg])
        cv.dry_now()
        self._sign(studio)

    def _sign(self, studio):
        """A small painted monogram and date on the table's front face, in a tone of the wood."""
        sc = self.sc
        cv, cam, rng, px = sc.cv, sc.cam, sc.rng, sc.px
        glyphs = {
            "C": [[(1, 0.82), (0.72, 1), (0.3, 1), (0.02, 0.74), (0, 0.26), (0.3, 0), (0.72, 0), (1, 0.2)]],
            "f": [[(0.85, 0.98), (0.55, 1.05), (0.38, 0.8), (0.34, 0)], [(0.1, 0.55), (0.7, 0.57)]],
            "2": [[(0.02, 0.76), (0.3, 1), (0.7, 1), (0.98, 0.76), (0.86, 0.5), (0.02, 0), (1, 0.02)]],
            "0": [[(0.5, 1), (0.12, 0.8), (0, 0.5), (0.12, 0.2), (0.5, 0), (0.88, 0.2), (1, 0.5), (0.88, 0.8), (0.5, 1)]],
            "6": [[(0.9, 0.95), (0.42, 0.9), (0.06, 0.5), (0.1, 0.16), (0.5, 0), (0.9, 0.18), (0.9, 0.42), (0.5, 0.58), (0.12, 0.42)]],
            ".": [[(0.1, 0.02), (0.2, 0.07)]], " ": []}
        from atelier.geom import catmull
        pen, y, size, z = 15.6, -0.92, 0.42, L.TABLE_FRONT + 0.01
        for ch in "C. f. 2026":
            for stroke in glyphs.get(ch, []):
                pts = np.array([[pen + a * size * 0.7 + rng.normal(0, 0.012), y + b * size + rng.normal(0, 0.012), z]
                                for a, b in stroke])
                xy = cam.project(pts)
                if len(xy) > 2:
                    xy = catmull(xy, max(1.0, px(1.6)))
                cv.stroke(xy, Paint.opaque("#7a5c3a"), Brush(radius=px(0.9), hardness=0.4, thick=0.55, pickup=0.2,
                          press_radius=0.6), np.clip(0.55 + 0.45 * np.sin(np.linspace(0.4, 2.8, len(xy))), 0, 1))
            pen += size * (0.42 if ch in " ." else 0.88)
        cv.dry_now()
