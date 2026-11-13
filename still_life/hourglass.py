"""The hourglass, knocked over and lying on its side.

Made as they were made around 1650: two separately blown bulbs, a little unlike each other,
butted neck to neck and bound at the waist with cloth and thread; red-painted parchment bands
where the domed ends sit in two octagonal oak boards; four hand-turned posts, blackened and worn
brown on their swellings. Lying down, the fine grey sand has run level in the low side of each bulb.
"""
import numpy as np

from atelier import forms
from atelier.brushwork import Batch, paint_surface
from atelier.canvas import Brush, Paint
from atelier.forms import Profile, fbm3, noise3
from atelier.geom import Ball, Capsule, Disc, Glow, normalize, smoothstep
from atelier.shading import lin, luma
from atelier.studio import Form
from . import scene as L

TWO_PI = 2 * np.pi
UP = np.array([0.0, 1.0, 0.0])

LENGTH = 19.0            # over the two boards
BOARD = 1.5              # thickness of each board
APOTHEM = 4.7            # centre of a board to the middle of one of its eight sides
POST_R = 3.75            # distance of the posts from the axis

OAK = lin("#5c4028")
OAK_WORN = lin("#86683f")
POST_BLACK = lin("#261b13")
SAND = lin("#a8957c")
BAND_RED = lin("#7a3526")
BINDING = lin("#8c6a4c")

# one bulb, from its domed end (0) to its neck: (distance along the axis, radius)
BULB = Profile([(0.0, 1.10), (0.45, 2.25), (1.5, 2.92), (3.0, 3.08), (4.6, 2.74), (5.9, 1.92), (6.8, 1.06),
                (7.3, 0.74), (7.75, 0.70)])
BULB_LEN = 7.75
SPINDLE = Profile([(0.0, 0.62), (0.5, 0.62), (0.75, 0.42), (1.1, 0.56), (1.45, 0.40), (4.0, 0.60), (7.0, 0.42),
                   (7.5, 0.40), (8.0, 0.58), (8.5, 0.40), (9.0, 0.42), (12.0, 0.60), (14.55, 0.40), (14.9, 0.56),
                   (15.25, 0.42), (15.5, 0.62), (16.0, 0.62)])


class Hourglass:
    softness = (0.20, 0.22)
    reach = 13.0

    def __init__(self, sc):
        self.sc = sc
        self.a = np.asarray(L.HOURGLASS, dtype=np.float64) * [1, 0, 1] + [0, APOTHEM, 0]   # centre of the near board's outer face
        self.d = normalize(L.HOURGLASS_AXIS)
        self.e1 = normalize(np.cross(self.d, UP))       # level, across the axis
        self.e2 = UP
        self.lt = None
        a, d = self.a, self.d
        # the two bulbs: not quite alike, not quite in line
        self.bulbs = []
        for k, (start, sign, tilt, fat) in enumerate(((BOARD + 0.05, 1, (0.012, -0.02), 1.0),
                                                      (LENGTH - BOARD - 0.05, -1, (-0.02, 0.015), 0.955))):
            base = a + start * d + 0.06 * (k - 0.5) * self.e1
            axis = normalize(sign * d + tilt[0] * self.e1 + tilt[1] * self.e2)
            surf = forms.roughened(forms.revolution(base, axis, lambda v, fat=fat: np.maximum(BULB(v), 0.0) * fat,
                                                    ref=UP), 0.05, 0.35, seed=k)
            self.bulbs.append(dict(base=base, axis=axis, surf=surf, fat=fat,
                                   level=APOTHEM - (0.75 if k == 0 else 1.55)))     # how high the sand lies in it
        self.waist_c = a + 0.5 * LENGTH * d
        self.posts = []
        for k, ang in enumerate(np.radians([48.0, 133.0, 226.0, 314.0])):
            rad = np.cos(ang) * self.e1 + np.sin(ang) * self.e2
            lean = 0.06 * np.array([np.sin(k * 2.1), np.cos(k * 1.3), 0.0])
            foot = a + BOARD * d + POST_R * rad
            axis = normalize(d + lean / 16.0)
            self.posts.append(dict(foot=foot, axis=axis, k=k,
                                   surf=forms.revolution(foot, axis, lambda v, k=k: SPINDLE(v) * (0.94 + 0.04 * np.sin(k * 1.7 + v)))))
        self.forms = []
        self._build()

    @property
    def centre(self):
        return self.a + 0.5 * LENGTH * self.d

    # ---- light
    def occluders(self):
        a, d = self.a, self.d
        occ = [Capsule("hg:near", a + 0.2 * d, a + (BOARD - 0.2) * d, APOTHEM * 0.98),
               Capsule("hg:far", a + (LENGTH - BOARD + 0.2) * d, a + (LENGTH - 0.2) * d, APOTHEM * 0.98)]
        for b in self.bulbs:
            occ.append(Ball("hg:glass", b["base"] + 3.2 * b["axis"], 2.95, opacity=0.28))
            occ.append(Ball("hg:sand", b["base"] + 3.2 * b["axis"] * [1, 0, 1] + [0, b["level"] - 1.3 - b["base"][1] * 0, 0]
                            if False else np.array([(b["base"] + 3.2 * b["axis"])[0], b["level"] - 0.9, (b["base"] + 3.2 * b["axis"])[2]]),
                            1.0, scale=(2.4, 0.9, 2.4)))
        for p in self.posts:
            occ.append(Capsule("hg:post%d" % p["k"], p["foot"], p["foot"] + 16.0 * p["axis"], 0.5))
        occ.append(Ball("hg:waist", self.waist_c, 1.1))
        return occ

    def glows(self):
        return [Glow("hg", self.a - 0.2 * self.d, 4.2, OAK), Glow("hg", self.bulbs[0]["base"] + 3.0 * self.bulbs[0]["axis"], 2.6, SAND * 0.8)]

    # ---- colours
    def _oak(self, P, along):
        """Dull waxed oak: open grain running one way, worn pale at the edges."""
        q = P @ np.stack([along, np.cross(along, self.d) if abs(along @ self.d) < 0.9 else self.e2, self.d], 1)
        grain = fbm3(np.stack([q[:, 0] * 0.5, q[:, 1] * 7.0, q[:, 2] * 7.0], 1), 1.0, 3, 61)
        pores = noise3(np.stack([q[:, 0] * 2.0, q[:, 1] * 40.0, q[:, 2] * 40.0], 1), 1.0, 63)
        return OAK * (0.72 + 0.50 * grain)[:, None] * (0.90 + 0.20 * pores)[:, None]

    def _in_octagon(self, P, centre, margin=0.0):
        p = P - centre
        u, v = p @ self.e1, p @ self.e2
        m = np.zeros(u.shape)
        for k in range(4):
            th = np.radians(-90.0 + 45.0 * k)
            m = np.maximum(m, np.abs(u * np.cos(th) + v * np.sin(th)))
        return m <= APOTHEM - margin, m

    def _build(self):
        sc = self.sc
        a, d, e1, e2 = self.a, self.d, self.e1, self.e2
        eye = sc.cam.eye

        def add(name, surf, ur, vr, colour, **kw):
            f = Form("hg_" + name, surf, ur, vr, colour, owner="hourglass", **kw)
            self.forms.append(f)
            return f

        def board(tag, s0, face_out):
            """One octagonal board: its eight edge faces and whichever flat face looks our way."""
            c0 = a + s0 * d
            skip = ("hg:" + tag,)
            half = APOTHEM * np.tan(np.radians(22.5))
            for k in range(8):
                th = np.radians(-90.0 + 45.0 * k)
                n = np.cos(th) * e1 + np.sin(th) * e2
                t = -np.sin(th) * e1 + np.cos(th) * e2
                origin = c0 + APOTHEM * n - half * t
                if (eye - (c0 + APOTHEM * n)) @ n <= 0:
                    continue                                         # this edge face looks away from us
                eu, ev = 2 * half * t, BOARD * d
                flip = np.cross(eu, ev) @ n < 0

                def col(P, N, u, v, t=t, skip=skip):
                    lt = self.lt or sc.lt
                    worn = np.maximum(smoothstep(0.85, 1.0, np.abs(2 * v - 1)), smoothstep(0.88, 1.0, np.abs(2 * u - 1)))
                    alb = self._oak(P, t) * (1 - 0.5 * worn[:, None]) + OAK_WORN * 0.5 * worn[:, None]
                    return lt.shade(alb, P, N, skip=skip, spec=0.05, gloss=10)
                add("%s_edge%d" % (tag, k), forms.plane(origin, eu, ev, flip=flip), (0, 1), (0, 1), col, size=3.0, grid=16,
                    along="u", flat=(origin, eu, ev), finish=0.7)
            # the flat face
            cf = c0 + (0.0 if face_out < 0 else BOARD) * d
            cf = c0 if (tag == "near") else c0          # near board: outer face at s0; far board: inner face at s0
            nrm = -d
            flip = np.cross(e1, e2) @ nrm < 0

            def face_col(P, N, u, v, cf=cf, skip=skip):
                lt = self.lt or sc.lt
                inside, m = self._in_octagon(P, cf)
                worn = smoothstep(APOTHEM - 0.45, APOTHEM, m)
                rings = 0.5 + 0.5 * np.sin(np.hypot((P - cf) @ e1, (P - cf) @ e2) * 5.2)      # faint turning marks
                alb = self._oak(P, e1) * (0.96 + 0.06 * rings)[:, None]
                alb = alb * (1 - 0.45 * worn[:, None]) + OAK_WORN * 0.45 * worn[:, None]
                return lt.shade(alb, P, N, skip=skip, spec=0.05, gloss=10)
            add(tag + "_face", forms.plane(cf, e1, e2, flip=flip), (-5.2, 5.2), (-5.2, 5.2), face_col, size=5.0, grid=30,
                vis=lambda P, N, u, v, cf=cf: self._in_octagon(P, cf)[0], along="u", flat=(cf, e1, e2),
                finish=0.9 if tag == "near" else 0.5, outline_grid=90)

        def post(p):
            k, foot = p["k"], p["foot"]
            ring = lambda v: np.clip(sum(np.exp(-((v - x) / 0.07) ** 2) for x in (0.75, 1.45, 7.5, 8.5, 14.55, 15.25)), 0, 1)

            def col(P, N, u, v):
                lt = self.lt or sc.lt
                # blackened, and rubbed back to brown wood wherever it swells out
                proud = smoothstep(0.47, 0.60, SPINDLE(v))
                alb = POST_BLACK * (0.8 + 0.5 * fbm3(P * 3.0, 1.0, 2, 71 + k))[:, None]
                alb = alb * (1 - 0.6 * proud[:, None]) + OAK * 0.9 * 0.6 * proud[:, None]
                alb = alb * (1 - 0.5 * ring(v))[:, None]
                return lt.shade(alb, P, N, skip=("hg:post%d" % k,), spec=0.12, gloss=24)
            add("post%d" % k, p["surf"], (0, TWO_PI), (0, 16.0), col, size=2.4, grid=70, along="v", finish=0.8)

        def sand_forms(i, b):
            base, axis, level = b["base"], b["axis"], b["level"]
            skip = ("hg:glass", "hg:sand")

            def sand_col(P, N, u, v):
                lt = self.lt or sc.lt
                alb = SAND * (0.92 + 0.14 * fbm3(P * 5.0, 1.0, 2, 81))[:, None]
                return lt.shade(alb, P, N, skip=skip, wrap=0.3, crowd=0.4)
            # the sand lying against the low side of the glass
            add("sand%d_side" % i, b["surf"], (0, TWO_PI), (0.0, BULB_LEN), sand_col, size=3.0, grid=60, along="v",
                vis=lambda P, N, u, v, level=level: P[..., 1] < level, finish=0.8, broken=0.02)

            # its level top, seen through the upper glass
            def top(u, v, base=base, axis=axis, level=level, fat=b["fat"]):
                u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
                C = base + u[..., None] * axis
                h = np.maximum(C[..., 1] - level, 0.0)
                r = np.maximum(BULB(u), 0.0) * fat
                w = np.sqrt(np.maximum(r * r - h * h, 0.0))
                P = C + (v * w)[..., None] * self.e1 - h[..., None] * UP
                return P, np.broadcast_to(UP, P.shape)
            add("sand%d_top" % i, top, (0.0, BULB_LEN), (-1.0, 1.0), sand_col, size=2.6, grid=50, along="u", finish=0.8,
                vis=lambda P, N, u, v, base=base, axis=axis, level=level, fat=b["fat"]:
                    (np.maximum(BULB(u), 0) * fat) > ((base + np.asarray(u)[..., None] * axis)[..., 1] - level) + 0.05,
                broken=0.02)

        def band(i, b):
            def col(P, N, u, v):
                lt = self.lt or sc.lt
                alb = BAND_RED * (0.75 + 0.5 * fbm3(P * 4.0, 1.0, 3, 91 + i))[:, None]
                return lt.shade(alb, P, N, skip=("hg:glass",), crowd=0.6)
            add("band%d" % i, b["surf"], (0, TWO_PI), (0.0, 1.35), col, size=2.4, grid=50, along="u", finish=0.6)

        def glass(i, b):
            # glass is not laid in like a solid thing; it is put on at the end, over what shows through it
            f = add("glass%d" % i, b["surf"], (0, TWO_PI), (1.35, BULB_LEN), lambda P, N, u, v: np.zeros((len(P), 3)),
                    size=3.0, grid=70, along="v", opaque=False, dead=False, finish=0.0, outline_grid=140)
            f.paint_me = False
            b["form"] = f

        def binding():
            c = self.waist_c
            surf = forms.roughened(forms.revolution(c - 0.85 * d, d, lambda v: 1.02 + 0.16 * np.sin(v * 3.6) ** 2), 0.07, 1.6, seed=5)

            def col(P, N, u, v):
                lt = self.lt or sc.lt
                thread = 0.5 + 0.5 * np.sin((v * 6.0 + u * 1.5) * 4.0)
                alb = BINDING * (0.70 + 0.45 * thread)[:, None] * (0.85 + 0.3 * fbm3(P * 6.0, 1.0, 2, 95))[:, None]
                return lt.shade(alb, P, N, skip=("hg:waist", "hg:glass"), crowd=0.7)
            add("waist", surf, (0, TWO_PI), (0, 1.7), col, size=2.0, grid=30, along="u", finish=0.8)

        depth = lambda P: sc.cam.depth(P)
        mid = depth(self.centre)
        board("far", LENGTH - BOARD, -1)
        for p in sorted(self.posts, key=lambda p: -depth(p["foot"] + 8 * p["axis"])):
            if depth(p["foot"] + 8 * p["axis"]) > mid:
                post(p)
        order = sorted(range(2), key=lambda i: -depth(self.bulbs[i]["base"] + 3.5 * self.bulbs[i]["axis"]))
        for i in order:
            b = self.bulbs[i]
            sand_forms(i, b)
            band(i, b)
            glass(i, b)
        binding()
        for p in sorted(self.posts, key=lambda p: -depth(p["foot"] + 8 * p["axis"])):
            if depth(p["foot"] + 8 * p["axis"]) <= mid:
                post(p)
        board("near", 0.0, -1)
        # forms that are only outlines to paint over later take no part in the ordinary stages
        for f in self.forms:
            if getattr(f, "paint_me", True) is False:
                f.finish = -1.0

    def groups(self):
        return [(self.sc.cam.depth(self.centre), self.forms)]

    def register(self, studio):
        studio.softness_of["hourglass"] = self.softness
        studio.hook("finish", self.glass)

    # ---- the glass itself
    def glass(self, studio):
        """Glass is painted by what it does to what is behind it and to the light: a breath of
        grey-green, deeper where the wall of the bulb is seen edge-on; the window caught once on
        each bulb; a thin broken line of light where the far table shows in its edge."""
        sc = self.sc
        cv, cam, rng, px = sc.cv, sc.cam, sc.rng, sc.px
        lt = self.lt or sc.lt
        tint = Paint.glaze("#5f6e58", 1.0)
        for i, b in enumerate(self.bulbs):
            f = b["form"]
            region = f.region
            if region.area() < 50:
                continue
            surf = f.surf

            def facing(P, N):
                return np.sum(N * cam.to_eye(P), axis=-1)
            common = dict(urange=f.ur, vrange=f.vr, along="v", stencil=region, grid=90)
            paint_surface(cv, cam, surf, None, Brush(radius=px(5), hardness=0.2, thick=0.07, pickup=0.0), rng,
                          length=px(40), spacing=px(4), paint=tint, **common)
            for lim, th in ((0.50, 0.16), (0.26, 0.34)):
                paint_surface(cv, cam, surf, None, Brush(radius=px(2.4), hardness=0.2, thick=th, pickup=0.0), rng,
                              length=px(24), spacing=px(2.2), paint=tint, margin=px(1.0),
                              visible=lambda P, N, u, v, lim=lim: facing(P, N) < lim, **common)
            cv.dry_now()

            # what the glass reflects: almost nothing head-on, more as it turns away
            def shine(P, N, u, v):
                c = np.clip(facing(P, N), 0, 1)
                fres = 0.045 + 0.75 * (1 - c) ** 4
                refl = lt.mirror(np.ones(3), P, N, skip=("hg:glass",), raw=True) * fres[:, None]
                return lt.tone(refl)

            def bright(P, N, u, v):
                shape = P.shape[:-1]
                col = shine(P.reshape(-1, 3), N.reshape(-1, 3), None, None)
                return (luma(col) > 0.13).reshape(shape)
            for r, sp in ((2.0, 1.6), (1.0, 0.9)):
                paint_surface(cv, cam, surf, shine, Brush(radius=px(r), hardness=0.5, thick=1.0, pickup=0.0), rng,
                              length=px(r * 3.5), spacing=px(sp), visible=bright, thick_by_value=(0.0, 2.0),
                              margin=px(0.6), coverage=1.3, **common)
            cv.dry_now()
        # dust of sand clinging inside the upper glass, and single grains catching light on the level sand
        speck = Batch(cv, Brush(radius=px(0.55), hardness=0.5, thick=0.5, pickup=0.1))
        for i, b in enumerate(self.bulbs):
            f = b["form"]
            for _ in range(260):
                u, v = rng.uniform(0, TWO_PI), rng.uniform(2.0, 7.4)
                P, N = f.surf(np.array([u]), np.array([v]))
                if P[0, 1] < b["level"] or np.sum(N[0] * cam.to_eye(P[0])) < 0.15:
                    continue
                xy = cam.project(P)[0]
                if not f.region.inside(xy[None], px(1.5))[0]:
                    continue
                c = lt.shade(SAND, P, N, skip=("hg:glass", "hg:sand"), wrap=0.4)[0]
                speck.add(np.array([xy, xy + rng.normal(0, px(0.6), 2)]), np.clip(c * rng.uniform(0.8, 1.2), 0, 1), 1.0,
                          rng.uniform(0.6, 1.3), tscale=rng.uniform(0.2, 0.7))
        speck.flush()
        cv.dry_now()


STUDIO_READY = True
