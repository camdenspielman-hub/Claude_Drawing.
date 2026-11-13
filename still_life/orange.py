"""Half an orange, cut side up at the front of the table.

A cut orange is never a diagram. Eleven segments of unequal size, their thin walls not quite
straight, meet in a ragged pithy core off the true centre; the flesh is thousands of juice sacs
lying outward from the core, each catching its own speck of light because it is wet; the pith is
a thick uneven band, and the skin outside it is dimpled and waxy. Two pips were cut through.
"""
import numpy as np

from atelier import forms
from atelier.brushwork import Batch, paint_surface
from atelier.canvas import Brush, Paint
from atelier.forms import fbm3, noise3
from atelier.geom import Ball, Glow, frame, normalize, smoothstep
from atelier.shading import lin
from atelier.studio import Form
from . import scene as L

TWO_PI = 2 * np.pi
PEEL = lin("#cf6412")
FLESH = lin("#ee8a16")
FLESH_DEEP = lin("#d2600c")
PITH = lin("#ecdcb4")
N_SEG = 11


class Orange:
    softness = (0.14, 0.0)
    reach = 9.0

    def __init__(self, sc):
        self.sc = sc
        self.lt = None
        self.R = L.ORANGE_R
        self.c = np.array([L.ORANGE[0], self.R * 0.985, L.ORANGE[2]])       # it rests on its rounded side
        self.n = normalize([-0.05, 0.985, 0.165])                            # the way the cut face looks: up, a little toward us
        self.e1, self.e2 = frame(self.n)
        rot = np.stack([self.e1, -self.n, self.e2], axis=1)
        self.dome = forms.roughened(forms.ellipsoid(self.c, (self.R, self.R * 0.97, self.R * 1.02), rot), 0.06, 0.5, seed=7)
        rng = np.random.default_rng(31)
        w = rng.uniform(0.5, 1.6, N_SEG)
        self.walls = np.concatenate([[0], np.cumsum(w)]) / w.sum() * TWO_PI + 0.4
        self.wall_bend = rng.normal(0, 0.06, N_SEG + 1)                      # each wall leans a little
        self.wall_curve = rng.normal(0, 0.14, N_SEG + 1)                     # and none is straight
        self.seg_tone = rng.uniform(0.84, 1.12, N_SEG)
        self.core = np.array([0.06, -0.04])                                  # the core is not at the true centre
        self.pips = [(self.walls[2] + 0.30, 0.27), (self.walls[7] + 0.22, 0.31)]

        def cut(u, v):          # u: angle, v: radius 0..1, with a slightly uneven rim and a face not quite flat
            u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
            rim = 1 + 0.018 * np.sin(3 * u + 1.0) + 0.012 * np.sin(7 * u)
            rr = self.R * v * rim
            x, y = rr * np.cos(u), rr * np.sin(u)
            sag = -0.05 * (1 - v ** 2) + 0.02 * np.sin(4 * u) * v                # the knife dragged; the middle has sunk a hair
            P = self.c + x[..., None] * self.e1 + y[..., None] * self.e2 + sag[..., None] * self.n
            return P, np.broadcast_to(self.n, P.shape)
        self.cut = cut
        self.forms = [
            Form("orange_peel", self.dome, (0, TWO_PI), (0, np.pi / 2), self._peel_col, owner="orange", size=5.0, grid=50,
                 along="u", tol=0.025, broken=0.04),
            Form("orange_cut", cut, (0, TWO_PI), (0, 1), self._face_col, owner="orange", size=3.6, grid=80, along="v",
                 tol=0.02, broken=0.05, outline_grid=140),
        ]

    @property
    def centre(self):
        return self.c

    def occluders(self):
        return [Ball("orange", self.c - 0.28 * self.R * self.n, 0.93 * self.R)]

    def glows(self):
        return [Glow("orange", self.c - 0.2 * self.R * self.n, self.R * 0.95, PEEL * 1.1),
                Glow("orange", self.c + 0.05 * self.n, self.R * 0.6, FLESH)]

    def groups(self):
        return [(self.sc.cam.depth(self.c + [0, -self.c[1], self.R * 0.7]), self.forms)]

    def register(self, studio):
        studio.softness_of["orange"] = self.softness
        studio.hook("model", self.sacs)
        studio.hook("finish", self.wet)

    # ---- the cut face, read in its own coordinates (radius 0..1 from the core, angle)
    def _face(self, P):
        d = P - self.c
        a, b = (d @ self.e1) / self.R - self.core[0], (d @ self.e2) / self.R - self.core[1]
        return np.hypot(a, b), np.mod(np.arctan2(b, a), TWO_PI), np.hypot((d @ self.e1) / self.R, (d @ self.e2) / self.R)

    def _parts(self, P):
        r, th, r_rim = self._face(P)
        dth = (th[:, None] - (self.walls + self.wall_bend * r[:, None] + self.wall_curve * np.sin(r[:, None] * 3.4))[..., :]
               + np.pi) % TWO_PI - np.pi
        wd = np.abs(dth).min(axis=1) * r
        seg = np.clip(np.searchsorted(self.walls, np.mod(th - 0.4, TWO_PI) + 0.4) - 1, 0, N_SEG - 1)
        ang = np.arctan2((P - self.c) @ self.e2, (P - self.c) @ self.e1)
        wav = 0.016 * np.sin(ang * 5 + 1.0) + 0.010 * np.sin(ang * 11) + 0.008 * np.sin(ang * 2 + 2.0)
        zest = smoothstep(0.952, 0.972, r_rim)
        pith = smoothstep(0.855 + wav, 0.885 + wav, r_rim) * (1 - zest)
        wall_w = 0.010 + 0.035 * smoothstep(0.26, 0.08, r) + 0.022 * smoothstep(0.78, 0.88, r_rim)
        wall = (1 - smoothstep(wall_w * 0.6, wall_w * 1.3, wd)) * (1 - pith) * (1 - zest)
        core = 1 - smoothstep(0.07, 0.125, r * (1 + 0.30 * np.cos(N_SEG * (th - 0.4)) + 0.2 * np.sin(3 * th)))
        return r, th, seg, zest, pith, np.maximum(wall, core), wd, r_rim

    def _face_col(self, P, N, u, v):
        lt = self.lt or self.sc.lt
        r, th, seg, zest, pith, white, wd, r_rim = self._parts(P)
        juice = noise3(np.stack([th * 30, r * 6, 0 * r], 1), 1.0, 111)
        deep = smoothstep(0.02, 0.16, wd) * smoothstep(0.12, 0.45, r)
        flesh = (FLESH * (1 - 0.55 * deep[:, None]) + FLESH_DEEP * 0.55 * deep[:, None]) \
            * (self.seg_tone[seg] * (0.82 + 0.36 * juice))[:, None]
        pith_col = PITH * (0.86 + 0.16 * noise3(P * 9.0, 1.0, 113))[:, None]
        # the pith drinks a little juice where it meets the flesh
        soak = (pith * smoothstep(0.93, 0.86, r_rim))[:, None]
        pith_col = pith_col * (1 - 0.35 * soak) + FLESH * 0.9 * 0.35 * soak
        alb = flesh * (1 - white[:, None]) + pith_col * white[:, None]
        alb = alb * (1 - pith[:, None]) + pith_col * pith[:, None]
        alb = alb * (1 - zest[:, None]) + PEEL * 0.95 * zest[:, None]
        hollow = 1 - 0.5 * (1 - smoothstep(0.0, 0.04, r))
        # the flesh is full of juice and lets light through: it never goes as dark as its shadow side
        glow = alb * 0.13 * (1 - white - pith - zest).clip(0, 1)[:, None]
        d = P - self.c
        out = d - (d @ self.n)[:, None] * self.n
        Nd = normalize(N + 0.30 * out / self.R)                  # the cut bulges a little: it turns from the light at its far side
        return lt.shade(alb * hollow[:, None] * 0.86, P, Nd, skip=("orange",), spec=0.04, gloss=20, emit=glow * 0.8, crowd=0.3)

    def _peel_col(self, P, N, u, v):
        lt = self.lt or self.sc.lt
        q = (P - self.c) * 8.0
        pores = fbm3(q, 1.0, 2, 121)
        eps = 0.05
        g = np.stack([fbm3(q + [eps, 0, 0], 1.0, 2, 121) - pores, fbm3(q + [0, eps, 0], 1.0, 2, 121) - pores,
                      fbm3(q + [0, 0, eps], 1.0, 2, 121) - pores], 1) / eps
        Nb = normalize(N - 0.12 * (g - np.sum(g * N, axis=1, keepdims=True) * N))      # the dimpled skin
        alb = PEEL * (0.80 + 0.35 * fbm3((P - self.c) * 1.3, 1.0, 3, 123))[:, None]
        alb = alb * (1 - 0.30 * smoothstep(0.55, 0.75, pores))[:, None]                # each oil gland a darker dot
        alb = alb * (1 - 0.22 * smoothstep(0.22, 0.0, v))[:, None]                     # a little bruised where it was cut
        return lt.shade(alb * 0.92, P, Nb, skip=("orange",), spec=0.22, gloss=20, floor_gain=1.25)

    def _flesh_only(self, P, N, u, v):
        shape = P.shape[:-1]
        r, th, seg, zest, pith, white, wd, r_rim = self._parts(P.reshape(-1, 3))
        return ((white < 0.3) & (pith < 0.2) & (zest < 0.2) & (r > 0.11)).reshape(shape)

    def sacs(self, studio):
        """The juice sacs: each a small stroke of its own colour, lying outward from the core."""
        sc = self.sc
        cv, cam, rng, px = sc.cv, sc.cam, sc.rng, sc.px
        f = studio.form("orange_cut")

        def sac(P, N, u, v):
            base = self._face_col(P, N, u, v)
            k = 0.78 + 0.5 * rng.random(len(P))
            return np.clip(base * np.stack([k, k ** 1.25, k ** 2.2], 1), 0, 1)       # lighter sacs go yellow, darker ones red
        paint_surface(cv, cam, self.cut, sac, Brush(radius=px(1.5), hardness=0.4, thick=0.9, pickup=0.45), rng,
                      urange=(0, TWO_PI), vrange=(0.1, 0.9), along="v", length=px(12), spacing=px(2.5), stencil=f.region,
                      margin=px(1.5), grid=120, visible=self._flesh_only, head=0.15, tail=0.45)

    def wet(self, studio):
        """What makes it wet: specks of light on the sacs turned to the window, two cut pips, the dark
        of juice soaked into the wood beside it."""
        sc = self.sc
        cv, cam, rng, px = sc.cv, sc.cam, sc.rng, sc.px
        lt = self.lt or sc.lt
        f = studio.form("orange_cut")
        centre = cam.project(self.c[None])[0]
        # the two pips, cut through: pale almond shapes with a darker rim
        for th, rr in self.pips:
            a = self.core[0] + rr * np.cos(th)
            b = self.core[1] + rr * np.sin(th)
            P = self.c + self.R * (a * self.e1 + b * self.e2)
            out = normalize(a * self.e1 + b * self.e2)
            tip0, tip1 = cam.project((P - 0.20 * out)[None])[0], cam.project((P + 0.20 * out)[None])[0]
            if f.region.inside(tip0[None])[0] and f.region.inside(tip1[None])[0]:
                cv.stroke(np.array([tip0, 0.5 * (tip0 + tip1), tip1]), Paint.opaque("#8a5a1e"),
                          Brush(radius=px(3.0), hardness=0.5, thick=1.2), [0.4, 1.0, 0.5])
                cv.stroke(np.array([tip0 * 0.8 + tip1 * 0.2, 0.5 * (tip0 + tip1), tip0 * 0.25 + tip1 * 0.75]), Paint.opaque("#eadfb6"),
                          Brush(radius=px(2.0), hardness=0.5, thick=1.4), [0.4, 1.0, 0.5])
        cv.dry_now()
        glint = Batch(cv, Brush(radius=px(0.9), hardness=0.6, thick=2.2, pickup=0.1, press_radius=0.6))
        toward = normalize(lt.light.L - (lt.light.L @ self.n) * self.n)
        lit_here = lt.light.lit(self.c[None], ("orange",))[0]
        # wet lights: gathered where the face turns the window back at us, a scattering elsewhere; never an even sprinkle
        wet_c = 0.42 * np.array([toward @ self.e1, toward @ self.e2])
        spots = np.concatenate([wet_c + rng.normal(0, 0.17, (26, 2)), rng.uniform(-0.9, 0.9, (14, 2))])
        for a_, b_ in spots:
            rr = float(np.hypot(a_, b_))
            if rr > 0.88 or rr < 0.12:
                continue
            P = self.c + self.R * (a_ * self.e1 + b_ * self.e2)
            if not self._flesh_only(P[None], None, None, None)[0] or rng.random() > lit_here:
                continue
            q = cam.project(P[None])[0]
            if not f.region.inside(q[None], px(1.0))[0]:
                continue
            out = (q - centre) / max(np.linalg.norm(q - centre), 1e-9)
            close = np.exp(-np.sum((np.array([a_, b_]) - wet_c) ** 2) / 0.05)
            ln = px(rng.uniform(1.5, 3.0) + 9.0 * close * rng.random())
            glint.add(np.array([q, q + out * ln]), np.array([1.0, 0.96, 0.84]) * rng.uniform(0.75, 1.0), [1.0, 0.4],
                      rng.uniform(0.6, 1.0) + 0.9 * close, tscale=0.4 + 0.6 * close)
        glint.flush()
        # juice on the table beside it: wood gone dark where it soaked in, with a thin wet shine
        table = studio.regions.get("table")
        if table is not None:
            spot = np.array([self.c[0] - 2.6, 0.0, self.c[2] + 3.4])
            for dx, dz, r in ((0.0, 0.0, 9.0), (0.7, 0.35, 6.0), (-0.6, 0.5, 4.0), (1.6, 0.1, 3.0)):
                q = cam.project((spot + [dx, 0, dz])[None])[0]
                if table.inside(q[None], px(r))[0]:
                    cv.stroke(np.array([q - [px(r), 0], q + [px(r), 0]]), Paint.glaze("#4a2a12", 1.0),
                              Brush(radius=px(r * 0.55), hardness=0.15, thick=0.45, pickup=0.0), 0.9)
            cv.dry_now()
            q = cam.project((spot + [0.2, 0, 0.1])[None])[0]
            if table.inside(q[None], px(6))[0]:
                cv.stroke(np.array([q - [px(5), px(0.5)], q + [px(4), -px(0.5)]]), Paint.opaque("#d9c8a2", body=1.5),
                          Brush(radius=px(1.0), hardness=0.3, thick=0.5, pickup=0.0, press_radius=0.6), [0.2, 1.0])
        cv.dry_now()


STUDIO_READY = True
