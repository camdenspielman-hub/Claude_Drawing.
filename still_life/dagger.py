"""The blade, driven point-first into the table: a dagger with a brass guard and pommel."""
import numpy as np

from atelier import forms
from atelier.brushwork import paint_form, paint_surface, line
from atelier.canvas import Brush, Paint
from atelier.forms import Curve, fbm3, noise3
from atelier.geom import Ball, Capsule, Quad, normalize, smoothstep
from atelier.shading import lin
from atelier.stencil import Stencil
from . import scene as L

TWO_PI = 2 * np.pi
LENGTH = 11.5        # of blade showing above the wood
BURIED = 1.2
RIDGE = 0.14         # half thickness along the middle of the blade
STEEL = np.array([0.74, 0.76, 0.78])
BRASS = np.array([0.95, 0.70, 0.30])
LEATHER = lin("#33211a")


def half_width(v):
    s = np.asarray(v, dtype=np.float64) + BURIED
    return 1.15 * np.minimum(1.0, (s / 4.5) ** 0.75) * (0.8 + 0.2 * s / (LENGTH + BURIED))


class Dagger:
    def __init__(self, sc):
        self.sc = sc
        self.e = np.asarray(L.BLADE, dtype=np.float64)
        self.a = normalize(L.BLADE_AXIS)
        a = self.a
        self.bw = normalize(np.cross(a, normalize([-0.25, 0.15, 0.95])))    # across the blade
        self.bn = np.cross(self.bw, a)                                      # the flat's own facing
        e, bw, bn = self.e, self.bw, self.bn

        def facet(sign):
            def surf(u, v):         # u: 0 at the cutting edge, 1 at the middle ridge; v: up the blade
                u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
                hw = half_width(v)
                P = e + v[..., None] * a + (sign * hw * (1 - u))[..., None] * bw + (RIDGE * u)[..., None] * bn
                N = normalize(hw[..., None] * bn + sign * RIDGE * bw + 0 * P)
                return P, N
            return surf
        self.facets = {"left": facet(-1), "right": facet(+1)}
        g = e + LENGTH * a
        self.g = g
        self.guard = forms.tube(Curve([g - 2.9 * bw, g, g + 2.9 * bw], smooth=False),
                                lambda v: 0.26 + 0.10 * np.abs(2 * v - 1) ** 2)
        self.finials = [forms.sphere(g - 2.95 * bw, 0.40), forms.sphere(g + 2.95 * bw, 0.40)]
        self.grip = forms.revolution(g + 0.28 * a, a, lambda v: 0.60 + 0.20 * np.sin(np.clip(v / 7.4, 0, 1) * np.pi) ** 0.8)
        self.pommel = forms.ellipsoid(g + 8.45 * a, (0.95, 0.85, 0.95), np.stack(forms.frame(a)[:1] + (a,) + forms.frame(a)[1:], axis=1))
        self.button = forms.sphere(g + 9.42 * a, 0.24)

    def occluders(self):
        e, a, bw, g = self.e, self.a, self.bw, self.g
        hw0, hw1 = half_width(0.0), half_width(LENGTH)
        return [Quad("dagger:blade", [e - hw0 * bw, e + hw0 * bw, g + hw1 * bw, g - hw1 * bw]),
                Capsule("dagger:guard", g - 2.9 * bw, g + 2.9 * bw, 0.32),
                Capsule("dagger:grip", g + 0.3 * a, g + 7.6 * a, 0.72),
                Ball("dagger:pommel", g + 8.45 * a, 0.92)]

    def ground_ao(self, P):
        d = np.hypot(P[:, 0] - self.e[0], P[:, 2] - self.e[2])
        return 1 - 0.5 * (1 - smoothstep(0.1, 1.2, d))

    def prepare(self):
        sc, cam = self.sc, self.sc.cam
        W, H = sc.W, sc.H
        st = {}
        self.above = lambda P, N, u, v: P[..., 1] > 0.0          # the wood hides the rest
        for k, f in self.facets.items():
            st[k] = Stencil.from_surface(cam, f, (0, 1), (-BURIED, LENGTH), W, H, grid=140, min_facing=-2.0,
                                         visible=self.above)
        st["guard"] = Stencil.from_surface(cam, self.guard, (0, TWO_PI), (0, 1), W, H, grid=90)
        for i, f in enumerate(self.finials):
            st["finial%d" % i] = Stencil.from_surface(cam, f, (0, TWO_PI), (-np.pi / 2, np.pi / 2), W, H, grid=28)
        st["grip"] = Stencil.from_surface(cam, self.grip, (0, TWO_PI), (0, 7.4), W, H, grid=110)
        st["pommel"] = Stencil.from_surface(cam, self.pommel, (0, TWO_PI), (-np.pi / 2, np.pi / 2), W, H, grid=60)
        st["button"] = Stencil.from_surface(cam, self.button, (0, TWO_PI), (-np.pi / 2, np.pi / 2), W, H, grid=20)
        self.st = st
        self.stencil = Stencil.empty(W, H)
        for s in st.values():
            self.stencil = self.stencil | s
        return self.stencil

    def paint(self):
        sc = self.sc
        cv, cam, lt, rng, px = sc.cv, sc.cam, sc.lt, sc.rng, sc.px
        if not hasattr(self, "st"):
            self.prepare()
        e, a, bw, bn, st = self.e, self.a, self.bw, self.bn, self.st
        orange_glow = lin("#e07818")

        # ---- the blade: two polished facets meeting along the ridge
        for name, facet in self.facets.items():
            def steel(P, N, u, v, name=name):
                # polished steel is never flat: the faint waves of the grinding bend its reflections into bands
                w1 = noise3(np.stack([v * 0.55, u * 0.6, 0 * v + (3.0 if name == "left" else 7.0)], 1), 1.0, 131) - 0.5
                w2 = noise3(np.stack([v * 2.4, u * 0.3, 0 * v], 1), 1.0, 133) - 0.5
                Nb = normalize(N + (0.22 * w1 + 0.06 * w2)[:, None] * a + (0.08 * w1)[:, None] * bw)
                spots = smoothstep(0.70, 0.90, fbm3(P * 6.0, 1.0, 3, 135))          # a few freckles of old rust
                col = lt.metal(STEEL, P, Nb, skip=("dagger:blade",), gloss=14.0, shine=1.5, env=1.0,
                               dirt=(lin("#4b3524"), 0.04 + 0.35 * spots), raw=True)
                # what a polished blade shows is the room: one facet turned to the light side of it, one to the dark,
                # each crossed by bands where the reflection slides from wall to window to table
                band = 0.5 + 0.5 * np.sin(v * 0.62 + (0.6 if name == "left" else 2.6) + 2.2 * w1)
                if name == "left":
                    col = col * (0.55 + 1.25 * smoothstep(0.25, 0.85, band))[:, None]
                    # the lit orange lies to this side: its colour comes back off the nearer facet, low down
                    col = col + orange_glow * (0.10 * smoothstep(5.0, 0.6, v) * smoothstep(0.0, 0.5, u))[:, None]
                else:
                    col = col * (0.22 + 0.75 * smoothstep(0.55, 0.95, band))[:, None]
                # the ground bevel along the cutting edge turns to the light
                col = col * (1 + 0.9 * smoothstep(0.10, 0.0, u))[:, None]
                return lt.tone(col)
            paint_form(sc, facet, (0, 1), (-BURIED, LENGTH), steel, st[name], r1=3.0, r2=1.4, along="v", grid=120,
                       blend=2.2, r_detail=0.9, tol=0.025, vis=self.above)
        cv.dry_now()
        # the ridge and the two edges, drawn with a fine point
        vs = np.linspace(0.05, LENGTH - 0.05, 80)
        ridge = e + vs[:, None] * a + RIDGE * bn
        line(cv, cam.project(ridge), Paint.opaque("#dfe2e2"), Brush(radius=px(0.7), hardness=0.6, thick=0.9, pickup=0.2), lo=0.4, hi=1.0)
        line(cv, cam.project(ridge) + [px(1.4), 0], Paint.opaque("#17181a"), Brush(radius=px(0.7), hardness=0.5, thick=0.6, pickup=0.2), lo=0.3, hi=0.8)
        left = e + vs[:, None] * a - half_width(vs)[:, None] * bw
        line(cv, cam.project(left) + [px(0.8), 0], Paint.opaque("#f1f2ee"), Brush(radius=px(0.75), hardness=0.6, thick=1.4, pickup=0.1), lo=0.3, hi=1.0)
        cv.dry_now()
        # one hard spark where the edge catches the window squarely
        spark = cam.project(e + 7.6 * a - half_width(7.6) * bw)
        cv.stroke(np.array([spark + [px(0.5), px(9)], spark + [px(0.5), -px(11)]]), Paint.opaque("#fffdf2"),
                  Brush(radius=px(1.2), hardness=0.5, thick=2.6, press_radius=0.7), [0.2, 1.0])

        # ---- the grip: dark leather bound with a spiral of cord
        def grip(P, N, u, v):
            cord = 0.5 + 0.5 * np.sin(v * 9.5 + u)
            alb = LEATHER * (0.55 + 0.75 * cord)[:, None] * (0.85 + 0.3 * fbm3(P * 4.0, 1.0, 2, 141))[:, None]
            Nb = normalize(N + (0.35 * np.cos(v * 9.5 + u))[:, None] * a)
            return lt.shade(alb, P, Nb, skip=("dagger:grip",), spec=0.16, gloss=18)
        paint_form(sc, self.grip, (0, TWO_PI), (0, 7.4), grip, st["grip"], r1=2.6, r2=1.2, along="u", grid=110,
                   blend=0, r_detail=0.9, tol=0.02)

        # ---- brass: pommel, button, guard and its two finials
        def brass(skip):
            def col(P, N, u, v):
                tar = smoothstep(0.35, 0.75, fbm3(P * 4.0, 1.0, 3, 151))
                return lt.metal(BRASS, P, N, skip=skip, gloss=10.0, shine=1.3, env=1.0, dirt=(lin("#5a4520"), 0.25 + 0.45 * tar))
            return col
        paint_form(sc, self.pommel, (0, TWO_PI), (-np.pi / 2, np.pi / 2), brass(("dagger:pommel",)), st["pommel"],
                   r1=2.8, r2=1.3, along="u", grid=60, blend=2.0)
        paint_form(sc, self.button, (0, TWO_PI), (-np.pi / 2, np.pi / 2), brass(("dagger:pommel",)), st["button"],
                   r1=1.4, r2=0, along="u", grid=20, blend=0, coverage=3.0)
        cv.dry_now()
        paint_form(sc, self.guard, (0, TWO_PI), (0, 1), brass(("dagger:guard",)), st["guard"], r1=2.2, r2=1.1,
                   along="v", grid=90, blend=1.6)
        for i, f in enumerate(self.finials):
            paint_form(sc, f, (0, TWO_PI), (-np.pi / 2, np.pi / 2), brass(("dagger:guard",)), st["finial%d" % i],
                       r1=1.8, r2=0.9, along="u", grid=28, blend=1.4, coverage=2.0)
        cv.dry_now()

        # ---- where it went into the oak: a dark slit, a lifted splinter, the grain opened along the plank
        def at_table(sign):                    # where one cutting edge meets the wood
            vs_ = np.linspace(-BURIED, 3.0, 400)
            pts = e + vs_[:, None] * a + (sign * half_width(vs_))[:, None] * bw
            return pts[np.argmin(np.abs(pts[:, 1]))] * [1, 0, 1]
        p_left, p_right = at_table(-1), at_table(+1)
        bt = normalize(p_right - p_left)
        hw = 0.5 * np.linalg.norm(p_right - p_left)
        e = 0.5 * (p_left + p_right)
        slit = np.array([p_left - bt * 0.12, p_right + bt * 0.12])
        line(cv, cam.project(slit) + [0, px(1.2)], Paint.opaque("#0b0705"), Brush(radius=px(1.5), hardness=0.5, thick=1.4), lo=0.6, hi=1.0)
        for sgn, ln in ((-1, 0.9), (1, 0.6)):
            xs = np.linspace(0, ln, 14)
            crack = e + bt * sgn * hw + np.stack([sgn * xs, 0 * xs, 0.10 * np.sin(xs * 9) * xs + 0.04 * xs], 1)
            line(cv, cam.project(crack), Paint.opaque("#120b07"), Brush(radius=px(0.8), hardness=0.5, thick=1.0, press_radius=0.7),
                 head=0.02, tail=0.9, lo=0.0, hi=1.0)
        splinter = np.array([e - bt * (hw + 0.05) + [0, 0.0, 0.10], e - bt * (hw + 0.45) + [0, 0.12, 0.16]])
        line(cv, cam.project(splinter), Paint.opaque("#b99667"), Brush(radius=px(0.8), hardness=0.6, thick=1.6), lo=0.5, hi=1.0)
        cv.dry_now()
