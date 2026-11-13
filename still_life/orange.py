"""Half an orange, tipped so the cut face looks at us: zest, pith, ten segments, wet flesh."""
import numpy as np

from atelier import forms
from atelier.brushwork import Batch, paint_surface, paint_form
from atelier.canvas import Brush
from atelier.forms import fbm3, noise3
from atelier.geom import Ball, frame, normalize, smoothstep
from atelier.shading import lin
from atelier.stencil import Stencil
from . import scene as L

TWO_PI = 2 * np.pi
PEEL = lin("#d86a12")
FLESH = lin("#f08a14")
FLESH_DEEP = lin("#d9650c")
PITH = lin("#f0deb4")
N_SEG = 10


class Orange:
    def __init__(self, sc):
        self.sc = sc
        self.c = np.asarray(L.ORANGE, dtype=np.float64)
        self.R = L.ORANGE_R
        self.n = normalize([-0.16, 0.84, 0.52])                  # the way the cut face looks
        self.e1, self.e2 = frame(self.n)
        rot = np.stack([self.e1, -self.n, self.e2], axis=1)      # the dome's own "up" is away from the cut
        self.dome = forms.ellipsoid(self.c, (self.R,) * 3, rot)
        self.cut = forms.disc(self.c, self.n, self.R)
        rng = np.random.default_rng(31)
        w = rng.uniform(0.8, 1.25, N_SEG)
        self.walls = np.concatenate([[0], np.cumsum(w)]) / w.sum() * TWO_PI + 0.4    # where the segment walls lie
        self.seg_tone = rng.uniform(0.86, 1.10, N_SEG)

    def occluders(self):
        return [Ball("orange", self.c - 0.22 * self.R * self.n, 0.95 * self.R)]

    def ground_ao(self, P):
        d = np.hypot(P[:, 0] - self.c[0], P[:, 2] - self.c[2])
        return 1 - 0.8 * (1 - smoothstep(0.4, 3.8, d))

    def prepare(self):
        sc, cam = self.sc, self.sc.cam
        self.st_dome = Stencil.from_surface(cam, self.dome, (0, TWO_PI), (0, np.pi / 2), sc.W, sc.H, grid=100)
        self.st_cut = Stencil.from_surface(cam, self.cut, (0, TWO_PI), (0, 1), sc.W, sc.H, grid=120)
        self.stencil = self.st_dome | self.st_cut
        return self.stencil

    # ---- the cut face, read in its own coordinates (radius 0..1, angle)
    def _face(self, P):
        d = P - self.c
        a, b = (d @ self.e1) / self.R, (d @ self.e2) / self.R
        return np.hypot(a, b), np.mod(np.arctan2(b, a), TWO_PI)

    def _wall_dist(self, r, th):
        """Distance (in radii) from the nearest segment wall, and which segment a point is in."""
        dth = np.abs((th[:, None] - self.walls[None, :] + np.pi) % TWO_PI - np.pi)
        seg = np.clip(np.searchsorted(self.walls, np.mod(th - 0.4, TWO_PI) + 0.4) - 1, 0, N_SEG - 1)
        return dth.min(axis=1) * r, seg

    def _parts(self, P):
        r, th = self._face(P)
        wd, seg = self._wall_dist(r, th)
        wav = 0.012 * np.sin(th * 7 + 1.0) + 0.008 * np.sin(th * 13)
        zest = smoothstep(0.955, 0.972, r)
        pith = smoothstep(0.885 + wav, 0.905 + wav, r) * (1 - zest)
        wall_w = 0.011 + 0.03 * smoothstep(0.25, 0.08, r) + 0.02 * smoothstep(0.82, 0.90, r)
        wall = (1 - smoothstep(wall_w * 0.6, wall_w * 1.3, wd)) * (1 - pith) * (1 - zest)
        core = (1 - smoothstep(0.075, 0.115, r * (1 + 0.25 * np.cos(N_SEG * (th - 0.4)))))
        return r, th, seg, zest, pith, np.maximum(wall, core), wd

    def _face_col(self, P, N, u, v):
        lt = self.sc.lt
        r, th, seg, zest, pith, white, wd = self._parts(P)
        # flesh: deeper in the middle of each segment, paler toward the walls and the centre
        juice = noise3(np.stack([th * 30, r * 6, 0 * r], 1), 1.0, 111)
        deep = smoothstep(0.02, 0.16, wd) * smoothstep(0.12, 0.45, r)
        flesh = (FLESH * (1 - 0.55 * deep[:, None]) + FLESH_DEEP * 0.55 * deep[:, None]) \
            * (self.seg_tone[seg] * (0.82 + 0.36 * juice))[:, None]
        alb = flesh * (1 - white[:, None]) + PITH * (0.9 + 0.1 * juice[:, None]) * white[:, None]
        alb = alb * (1 - pith[:, None]) + PITH * pith[:, None]
        alb = alb * (1 - zest[:, None]) + PEEL * 0.95 * zest[:, None]
        hollow = 1 - 0.5 * (1 - smoothstep(0.0, 0.035, r))
        # the flesh is full of juice and lets light through: it never goes as dark as its shadow side
        glow = alb * 0.14 * (1 - white - pith - zest).clip(0, 1)[:, None]
        return lt.shade(alb * hollow[:, None], P, N, skip=("orange",), spec=0.05, gloss=20, emit=glow)

    def _peel_col(self, P, N, u, v):
        lt = self.sc.lt
        q = (P - self.c) * 9.0
        pores = fbm3(q, 1.0, 2, 121)
        eps = 0.05
        g = np.stack([fbm3(q + [eps, 0, 0], 1.0, 2, 121) - pores, fbm3(q + [0, eps, 0], 1.0, 2, 121) - pores,
                      fbm3(q + [0, 0, eps], 1.0, 2, 121) - pores], 1) / eps
        Nb = normalize(N - 0.10 * (g - np.sum(g * N, axis=1, keepdims=True) * N))      # the pebbled skin
        alb = PEEL * (0.80 + 0.35 * fbm3((P - self.c) * 1.3, 1.0, 3, 123))[:, None] * (1 - 0.25 * smoothstep(0.55, 0.75, pores))[:, None]
        rim = 1 - 0.25 * smoothstep(0.25, 0.0, v)[:, None]                              # a little bruised where it was cut
        return lt.shade(alb * rim, P, Nb, skip=("orange",), spec=0.28, gloss=22, floor_gain=1.6)

    def paint(self):
        sc = self.sc
        cv, cam, lt, rng, px = sc.cv, sc.cam, sc.lt, sc.rng, sc.px
        if not hasattr(self, "st_cut"):
            self.prepare()
        # the skin
        paint_form(sc, self.dome, (0, TWO_PI), (0, np.pi / 2), self._peel_col, self.st_dome, r1=5, r2=2.2, along="u",
                   grid=60, blend=2.5, tol=0.025)
        cv.dry_now()
        # the cut face: broad lay-in from the centre outward, then the small brush finds the walls and the pith
        paint_form(sc, self.cut, (0, TWO_PI), (0, 1), self._face_col, self.st_cut, r1=4, r2=1.8, along="v", grid=90,
                   blend=0, r_detail=1.0, tol=0.02)
        paint_surface(cv, cam, self.cut, self._face_col, Brush(radius=px(0.9), hardness=0.6, thick=1.2, pickup=0.5), rng,
                      urange=(0, TWO_PI), vrange=(0, 1), along="v", length=px(4), spacing=px(0.9), stencil=self.st_cut,
                      margin=px(0.5), grid=120, detail=(0.02, px(2.2)))
        cv.dry_now()

        # the juice sacs: each a small stroke of its own colour, running out from the centre
        def flesh_only(P, N, u, v):
            shape = P.shape[:-1]
            r, th, seg, zest, pith, white, wd = self._parts(P.reshape(-1, 3))
            return ((white < 0.3) & (pith < 0.2) & (zest < 0.2) & (r > 0.1)).reshape(shape)

        def sac(P, N, u, v):
            base = self._face_col(P, N, u, v)
            k = 0.78 + 0.5 * rng.random(len(P))
            warm = np.stack([k, k ** 1.25, k ** 2.2], 1)       # lighter sacs go yellow, darker ones red-orange
            return np.clip(base * warm, 0, 1)
        paint_surface(cv, cam, self.cut, sac, Brush(radius=px(1.5), hardness=0.4, thick=0.9, pickup=0.45), rng,
                      urange=(0, TWO_PI), vrange=(0.1, 0.9), along="v", length=px(13), spacing=px(2.6),
                      stencil=self.st_cut, margin=px(2), grid=120, visible=flesh_only, head=0.15, tail=0.45)
        cv.dry_now()

        # wet lights: the cut flesh glistens where it tilts to the window
        centre = cam.project(self.c)
        glint = Batch(cv, Brush(radius=px(0.9), hardness=0.6, thick=2.2, pickup=0.1, press_radius=0.6))
        for _ in range(520):
            rr, th = 0.12 + 0.76 * np.sqrt(rng.random()), rng.uniform(0, TWO_PI)
            P = self.c + self.R * rr * (np.cos(th) * self.e1 + np.sin(th) * self.e2)
            if not flesh_only(P[None], None, None, None)[0]:
                continue
            # a band of sparkle across the face, strongest on the side toward the light
            side = (P - self.c) @ normalize(lt.light.L - (lt.light.L @ self.n) * self.n) / self.R
            if rng.random() > 0.25 + 0.6 * smoothstep(-0.6, 0.7, side):
                continue
            q = cam.project(P)
            out = (q - centre) / max(np.linalg.norm(q - centre), 1e-9)
            ln = px(rng.uniform(1.0, 3.5))
            glint.add(np.array([q, q + out * ln]), np.array([1.0, 0.97, 0.86]) * rng.uniform(0.85, 1.0), [1.0, 0.4],
                      rng.uniform(0.6, 1.3))
        glint.flush()
        cv.dry_now()
