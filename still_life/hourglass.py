"""The hourglass: two turned wooden plates, three spindles, a waisted glass, and the sand nearly run out."""
import numpy as np

from atelier import forms
from atelier.brushwork import paint_surface, paint_form, line
from atelier.canvas import Brush, Paint
from atelier.forms import Profile, fbm3, noise3
from atelier.geom import Ball, Capsule, Disc, normalize, smoothstep
from atelier.shading import lin
from atelier.stencil import Stencil
from . import scene as L

UP = np.array([0.0, 1.0, 0.0])
TWO_PI = 2 * np.pi

HEIGHT = 16.0
PLATE = 0.95                 # thickness of each wooden plate
R_PLATE = 4.5
POST_R = 3.55                # distance of the spindles from the axis
POST_ANGLES = np.radians([47.0, 167.0, 287.0])     # the last one stands behind the glass

BASE_EDGE = Profile([(0.0, 4.30), (0.12, 4.50), (0.42, 4.50), (0.55, 4.32), (0.70, 4.40), (0.95, 4.25)])
TOP_EDGE = Profile([(15.05, 4.25), (15.30, 4.40), (15.45, 4.32), (15.58, 4.50), (15.88, 4.50), (16.0, 4.30)])
GLASS = Profile([(1.0, 1.30), (1.25, 2.20), (2.0, 2.80), (3.3, 3.00), (4.6, 2.75), (5.8, 2.10), (6.9, 1.10),
                 (7.6, 0.42), (8.0, 0.26), (8.4, 0.42), (9.1, 1.10), (10.2, 2.10), (11.4, 2.75), (12.7, 3.00),
                 (14.0, 2.80), (14.75, 2.20), (15.0, 1.30)])
SPINDLE = Profile([(0.0, 0.55), (0.4, 0.55), (0.6, 0.38), (0.9, 0.50), (1.2, 0.36), (1.6, 0.42), (3.2, 0.58),
                   (5.2, 0.36), (5.9, 0.33), (6.3, 0.50), (6.7, 0.36), (7.05, 0.52), (7.4, 0.36), (7.8, 0.50),
                   (8.2, 0.33), (8.9, 0.36), (10.9, 0.58), (12.5, 0.42), (12.9, 0.36), (13.2, 0.50), (13.5, 0.38),
                   (13.7, 0.55), (14.1, 0.55)])

NECK = 8.0
SAND_LOW = 4.7               # level of the sand against the glass, lower bulb
SAND_PEAK = 5.9              # top of the little cone it falls onto
SAND_HIGH = 10.0             # what is left above: level against the glass
SAND_PIT = 8.5               # bottom of the funnel it drains through

WOOD = lin("#55321a")
WOOD_DARK = lin("#2c170b")
SAND = lin("#d3c09a")


class Hourglass:
    def __init__(self, sc, pos=None):
        self.sc = sc
        self.c = np.asarray(L.HOURGLASS if pos is None else pos, dtype=np.float64)
        c = self.c
        self.glass = forms.revolution(c, UP, lambda v: np.maximum(GLASS(v), 0.0))
        self.base_side = forms.revolution(c, UP, BASE_EDGE)
        self.top_side = forms.revolution(c, UP, TOP_EDGE)
        self.base_top = forms.plane(c + [0, PLATE, 0], [1.0, 0, 0], [0, 0, 1.0], flip=True)
        self.top_top = forms.plane(c + [0, HEIGHT, 0], [1.0, 0, 0], [0, 0, 1.0], flip=True)
        self.post_feet = [c + [POST_R * np.cos(a), PLATE, POST_R * np.sin(a)] for a in POST_ANGLES]
        self.posts = [forms.revolution(f, UP, SPINDLE) for f in self.post_feet]
        r_low, r_high = float(GLASS(SAND_LOW)), float(GLASS(SAND_HIGH))

        def mound(u, t):          # the cone of fallen sand: t = 0 at the peak, 1 at the glass
            u, t = np.asarray(u, dtype=np.float64), np.asarray(t, dtype=np.float64)
            rad = np.stack([np.cos(u), 0 * u, np.sin(u)], -1)
            P = c + rad * (r_low * t)[..., None] + UP * (SAND_PEAK - (SAND_PEAK - SAND_LOW) * t ** 0.9)[..., None]
            slope = (SAND_PEAK - SAND_LOW) / r_low
            return P, normalize(rad * slope + UP)

        def funnel(u, t):         # the pit in the sand that is left: t = 0 at the bottom, 1 at the glass
            u, t = np.asarray(u, dtype=np.float64), np.asarray(t, dtype=np.float64)
            rad = np.stack([np.cos(u), 0 * u, np.sin(u)], -1)
            P = c + rad * (0.12 + (r_high - 0.12) * t)[..., None] + UP * (SAND_PIT + (SAND_HIGH - SAND_PIT) * t)[..., None]
            slope = (SAND_HIGH - SAND_PIT) / r_high
            return P, normalize(-rad * slope + UP)
        self.mound, self.funnel = mound, funnel

    # ---- what it blocks of the light
    def occluders(self):
        c = self.c
        occ = [Disc("hg:base", c + [0, 0.5, 0], UP, R_PLATE), Disc("hg:top", c + [0, 15.55, 0], UP, R_PLATE),
               Ball("hg:glass", c + [0, 3.6, 0], 2.95, opacity=0.30), Ball("hg:glass", c + [0, 12.4, 0], 2.95, opacity=0.30),
               Ball("hg:sand", c + [0, 3.2, 0], 1.0, scale=(2.75, 2.3, 2.75)),
               Ball("hg:sand", c + [0, 9.2, 0], 1.0, scale=(1.5, 1.0, 1.5))]
        for i, f in enumerate(self.post_feet):
            occ.append(Capsule("hg:post%d" % i, f, f + [0, 14.1, 0], 0.44))
        return occ

    def ground_ao(self, P):
        d = np.hypot(P[:, 0] - self.c[0], P[:, 2] - self.c[2])
        return 1 - 0.7 * (1 - smoothstep(4.3, 6.2, d))

    # ---- outlines
    def _disc_vis(self, centre_y, radius):
        c = self.c

        def vis(P, N, u, v):
            return np.hypot(P[..., 0] - c[0], P[..., 2] - c[2]) < radius
        return vis

    def prepare(self):
        sc, cam = self.sc, self.sc.cam
        W, H = sc.W, sc.H
        st = {}
        st["base_side"] = Stencil.from_surface(cam, self.base_side, (0, TWO_PI), (0, PLATE), W, H, grid=120)
        st["base_top"] = Stencil.from_surface(cam, self.base_top, (-4.4, 4.4), (-4.4, 4.4), W, H, grid=120,
                                              visible=self._disc_vis(PLATE, 4.25))
        st["top_side"] = Stencil.from_surface(cam, self.top_side, (0, TWO_PI), (15.05, HEIGHT), W, H, grid=120)
        st["top_top"] = Stencil.from_surface(cam, self.top_top, (-4.4, 4.4), (-4.4, 4.4), W, H, grid=120,
                                             visible=self._disc_vis(HEIGHT, 4.30))
        st["glass"] = Stencil.from_surface(cam, self.glass, (0, TWO_PI), (1.0, 15.0), W, H, grid=160)
        for i, p in enumerate(self.posts):
            st["post%d" % i] = Stencil.from_surface(cam, p, (0, TWO_PI), (0, 14.1), W, H, grid=120)
        self.st = st
        self.stencil = Stencil.empty(W, H)
        self.reserve = Stencil.empty(W, H)      # the glass is see-through: the background goes on behind it
        for k, s in st.items():
            self.stencil = self.stencil | s
            if k != "glass":
                self.reserve = self.reserve | s
        return self.stencil

    # ---- colours
    def _wood(self, P, turned_about=None):
        """Dark polished walnut. For turned parts the grain shows as bands round the axis."""
        q = (P - self.c) * [2.2, 0.35, 2.2] if turned_about is None else np.stack(
            [np.hypot(P[:, 0] - turned_about[0], P[:, 2] - turned_about[2]) * 3.0, P[:, 1] * 6.0, 0 * P[:, 0]], 1)
        g = fbm3(q, 1.0, 3, 61)
        f = noise3(q * [1, 9, 1], 1.0, 63)
        a = WOOD * (0.70 + 0.55 * g)[:, None] * (0.92 + 0.16 * f)[:, None]
        dark = smoothstep(0.55, 0.8, fbm3(q * 0.5 + 4.0, 1.0, 2, 65))[:, None]
        return a * (1 - 0.4 * dark) + WOOD_DARK * 0.4 * dark

    def _wood_col(self, skip, turned_about=None, groove=None):
        lt = self.sc.lt

        def col(P, N, u, v):
            a = self._wood(P, turned_about)
            if groove is not None:
                a = a * (1 - 0.55 * groove(v))[:, None]
            return lt.shade(a, P, N, skip=skip, spec=0.22, gloss=36)
        return col

    # ---- painting
    def paint(self):
        sc = self.sc
        cv, cam, lt, rng, px = sc.cv, sc.cam, sc.lt, sc.rng, sc.px
        if not hasattr(self, "st"):
            self.prepare()
        st, c = self.st, self.c

        # the base plate
        paint_form(sc, self.base_top, (-4.4, 4.4), (-4.4, 4.4), self._wood_col(("hg:base",)), st["base_top"],
                   r1=5, r2=2.2, along="u", vis=self._disc_vis(PLATE, 4.25), grid=60)
        groove = lambda v: np.exp(-((v - 0.52) / 0.05) ** 2)
        paint_form(sc, self.base_side, (0, TWO_PI), (0, PLATE), self._wood_col(("hg:base",), c, groove), st["base_side"],
                   r1=3.2, r2=1.6, along="u", grid=80, blend=2.0)
        cv.dry_now()

        # the spindle behind the glass
        self._post(2)
        cv.dry_now()

        # ---- inside the glass
        g_ur = (0, TWO_PI)
        sand_skip = ("hg:glass", "hg:sand")

        def sand_col(P, N, u, v):
            q = (P - c) * 16.0
            a = SAND * (0.86 + 0.26 * fbm3(q, 1.0, 2, 71))[:, None]
            return lt.shade(a, P, N, skip=sand_skip, wrap=0.25)
        gst = st["glass"]
        # lower bulb: the sand lying against the glass, then the cone it falls onto
        paint_form(sc, self.glass, g_ur, (1.0, SAND_LOW), sand_col, gst, r1=4, r2=2, along="u", grid=80, blend=2.5,
                   edge=False)
        paint_form(sc, self.mound, g_ur, (0.0, 1.0), sand_col, gst, r1=3, r2=1.6, along="v", grid=60, blend=2.0,
                   edge=False)
        # upper bulb: the far wall of the draining pit, then the little sand left against the glass
        paint_form(sc, self.funnel, g_ur, (0.0, 1.0), sand_col, gst, r1=2.4, r2=1.4, along="v", grid=60, blend=2.0,
                   edge=False)
        paint_form(sc, self.glass, g_ur, (NECK + 0.05, SAND_HIGH), sand_col, gst, r1=3, r2=1.6, along="u", grid=80,
                   blend=2.0, edge=False)
        cv.dry_now()
        # grain: a stipple of single lighter and darker grains
        self._sand_grain(sand_col)
        # the falling thread of sand
        fall = np.stack([np.full(24, c[0]), np.linspace(NECK - 0.15, SAND_PEAK + 0.05, 24), np.full(24, c[2])], 1)
        line(cv, cam.project(fall) + rng.normal(0, px(0.25), (24, 2)), Paint.opaque("#d8c7a2"),
             Brush(radius=px(1.0), hardness=0.6, thick=1.0, pickup=0.3), lo=0.7, hi=1.0)
        cv.dry_now()

        self._glass()
        cv.dry_now()

        # the two spindles in front, then the top plate over everything
        self._post(1)
        self._post(0)
        cv.dry_now()
        groove = lambda v: np.exp(-((v - 15.47) / 0.05) ** 2)
        paint_form(sc, self.top_side, (0, TWO_PI), (15.05, HEIGHT), self._wood_col(("hg:top",), c, groove), st["top_side"],
                   r1=3.2, r2=1.6, along="u", grid=80, blend=2.0)
        paint_form(sc, self.top_top, (-4.4, 4.4), (-4.4, 4.4), self._wood_col(("hg:top",)), st["top_top"],
                   r1=5, r2=2.2, along="u", vis=self._disc_vis(HEIGHT, 4.30), grid=60)
        cv.dry_now()

    def _post(self, i):
        sc = self.sc
        foot = self.post_feet[i]
        ring = lambda v: np.clip(sum(np.exp(-((v - x) / 0.06) ** 2) for x in (0.6, 1.2, 5.9, 6.7, 7.4, 8.2, 12.9, 13.5)), 0, 1)
        paint_form(sc, self.posts[i], (0, TWO_PI), (0, 14.1), self._wood_col(("hg:post%d" % i,), foot, ring),
                   self.st["post%d" % i], r1=2.6, r2=1.3, along="v", grid=110, blend=1.8, r_detail=1.0)

    def _sand_grain(self, sand_col):
        sc = self.sc
        cv, cam, rng, px = sc.cv, sc.cam, sc.rng, sc.px
        for surf, vr in ((self.glass, (1.0, SAND_LOW)), (self.mound, (0.0, 1.0)), (self.glass, (NECK + 0.1, SAND_HIGH))):
            for tone in (1.16, 0.80):
                def col(P, N, u, v, tone=tone):
                    return np.clip(sand_col(P, N, u, v) * tone, 0, 1)
                paint_surface(cv, cam, surf, col, Brush(radius=px(0.8), hardness=0.6, thick=0.8, pickup=0.2), rng,
                              urange=(0, TWO_PI), vrange=vr, length=px(1.6), spacing=px(4.0), stencil=self.st["glass"],
                              margin=px(1.5), grid=60, npts=3, coverage=0.45)
        cv.dry_now()

    def _silhouette(self, v0, v1, n=120):
        """The glass's left and right outlines as canvas polylines (where it turns away from the eye)."""
        cam = self.sc.cam
        vs = np.linspace(v0, v1, n)
        us = np.linspace(0, TWO_PI, 721)
        U, V = np.meshgrid(us, vs, indexing="ij")
        P, N = self.glass(U, V)
        f = np.sum(N * cam.to_eye(P), axis=-1)
        xy = cam.project(P)
        left, right = [], []
        for j in range(n):
            cross = np.nonzero(np.sign(f[:-1, j]) != np.sign(f[1:, j]))[0]
            if len(cross) < 2:
                continue
            pts = xy[cross, j]
            left.append(pts[np.argmin(pts[:, 0])])
            right.append(pts[np.argmax(pts[:, 0])])
        return np.array(left), np.array(right)

    def _glass(self):
        """Glass is painted by what it does to light: a faint tint, darker turning edges,
        the window caught on each bulb, and its dimmer echo on the far inside wall."""
        sc = self.sc
        cv, cam, lt, rng, px = sc.cv, sc.cam, sc.lt, sc.rng, sc.px
        gst, c = self.st["glass"], self.c
        g_ur, g_vr = (0, TWO_PI), (1.0, 15.0)
        tint = Paint.glaze("#6f7d68", 1.0)

        def facing(P, N):
            return np.sum(N * cam.to_eye(P), axis=-1)
        # a breath of grey-green over the whole glass, then more where it turns edge-on
        paint_surface(cv, cam, self.glass, None, Brush(radius=px(5), hardness=0.3, thick=0.10, pickup=0.0), rng,
                      urange=g_ur, vrange=g_vr, along="v", length=px(40), spacing=px(4), stencil=gst, grid=90, paint=tint)
        for lim, th in ((0.55, 0.22), (0.30, 0.40)):
            paint_surface(cv, cam, self.glass, None, Brush(radius=px(2.6), hardness=0.3, thick=th, pickup=0.0), rng,
                          urange=g_ur, vrange=g_vr, along="v", length=px(26), spacing=px(2.4), stencil=gst, margin=px(1.0),
                          grid=120, paint=tint, visible=lambda P, N, u, v, lim=lim: facing(P, N) < lim)
        cv.dry_now()

        # the room, dimly: the faintest cool veil on the side toward the window
        def veil(P, N, u, v):
            return np.sum(N * lt.light.L, axis=-1) > 0.62
        paint_surface(cv, cam, self.glass, None, Brush(radius=px(16), hardness=0.0, thick=0.10, pickup=0.0), rng,
                      urange=g_ur, vrange=g_vr, along="v", length=px(120), spacing=px(3), stencil=gst, margin=px(8),
                      grid=120, paint=Paint.opaque("#aeb9c0", body=0.07), visible=veil)
        cv.dry_now()

        # outlines: light along the window side, dark along the other
        for v0, v1 in ((1.3, 7.7), (8.3, 14.7)):
            left, right = self._silhouette(v0, v1)
            if len(left) > 3:
                line(cv, left + [px(1.6), 0], Paint.opaque("#c9cdc6"), Brush(radius=px(0.9), hardness=0.5, thick=0.7, pickup=0.2),
                     lo=0.25, hi=0.9)
                line(cv, right - [px(1.4), 0], Paint.opaque("#14100c"), Brush(radius=px(1.1), hardness=0.5, thick=0.6, pickup=0.2),
                     lo=0.3, hi=0.9)
                line(cv, right - [px(4.5), 0], Paint.opaque("#8f958c"), Brush(radius=px(0.8), hardness=0.4, thick=0.35, pickup=0.2),
                     lo=0.1, hi=0.7)
        cv.dry_now()

        # the window, caught on the curve of each bulb
        Lw = lt.light.L
        for yc in (3.6, 12.4):
            V = cam.to_eye(c + [0, yc, 0])
            Hv = normalize(Lw + V)
            t1 = normalize(np.cross(UP, Hv))
            t2 = np.cross(Hv, t1)
            lo, hi = (1.2, 7.4) if yc < 8 else (8.6, 14.8)
            for sign, wa, wb, gap, tone, thick in ((1, 0.115, 0.165, 0.016, "#fbf5e2", 2.4), (-1, 0.15, 0.21, 0.02, "#a9a89a", 0.6)):
                def pane(P, N, u, v, sign=sign, wa=wa, wb=wb, gap=gap):
                    d = np.sum(N * Hv, axis=-1) * sign
                    a = np.sum(N * t1, axis=-1) / np.maximum(d, 1e-6)
                    b = np.sum(N * t2, axis=-1) / np.maximum(d, 1e-6)
                    return (d > 0.5) & (np.abs(a) < wa) & (np.abs(b) < wb) & (np.abs(a) > gap) & (np.abs(b) > gap)
                # a soft bloom first, then the panes themselves, thick and light
                if sign > 0:
                    def bloom(P, N, u, v):
                        d = np.sum(N * Hv, axis=-1)
                        a = np.sum(N * t1, axis=-1) / np.maximum(d, 1e-6)
                        b = np.sum(N * t2, axis=-1) / np.maximum(d, 1e-6)
                        return (d > 0.5) & (np.abs(a) < wa * 1.7) & (np.abs(b) < wb * 1.5)
                    paint_surface(cv, cam, self.glass, None, Brush(radius=px(4.0), hardness=0.05, thick=0.10, pickup=0.0), rng,
                                  urange=g_ur, vrange=(lo, hi), along="v", length=px(10), spacing=px(2.0), grid=160,
                                  paint=Paint.opaque("#e9e6d6", body=0.5), visible=bloom, stencil=gst)
                paint_surface(cv, cam, self.glass, None, Brush(radius=px(1.3), hardness=0.6, thick=thick, pickup=0.0), rng,
                              urange=g_ur, vrange=(lo, hi), along="v", length=px(5), spacing=px(1.1), grid=200,
                              paint=Paint.opaque(tone), visible=pane, min_facing=-2.0 if sign < 0 else 0.02, npts=4,
                              coverage=2.0, stencil=gst)
        # a spark at the waist
        neck = cam.project(c + [-0.2, NECK + 0.25, 0.2])
        cv.stroke(np.array([neck, neck + [px(1.5), px(5)]]), Paint.opaque("#f4efe0"),
                  Brush(radius=px(1.1), hardness=0.6, thick=1.8), [0.4, 1.0])
