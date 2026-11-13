"""The ball of yarn with its two needles, and the strand that runs from it across the table.

Yarn suits a brush: every strand is a stroke. Each is laid three times - a dark bed, the body
of the wool, a lighter touch along the side that faces the window - then crossed with the small
slanting marks of its twist and broken with stray fibres.
"""
import numpy as np

from atelier import forms
from atelier.brushwork import Batch, paint_form, line
from atelier.canvas import Brush, Paint
from atelier.forms import Curve, fbm3, noise3
from atelier.geom import Ball, Capsule, catmull, normalize, resample, smoothstep
from atelier.shading import lin
from atelier.stencil import Stencil
from . import scene as L

TWO_PI = 2 * np.pi
UP = np.array([0.0, 1.0, 0.0])
STRAND = 0.17                       # radius of one strand of yarn, cm
WOOL = lin("#4a7f99")
WOOL_DEEP = lin("#22404f")
NEEDLE = lin("#d2c39c")


def wool_albedo(P, seed=0):
    """Blue-green wool, unevenly dyed."""
    n = fbm3(P * 0.9 + seed * 3.1, 1.0, 3, 91)[:, None]
    h = noise3(P * 3.0 + seed, 1.0, 93)[:, None]
    return (WOOL * (1 - 0.45 * smoothstep(0.35, 0.75, n)) + WOOL_DEEP * 0.45 * smoothstep(0.35, 0.75, n)) * (0.85 + 0.3 * h)


def paint_strands(sc, strands, skip=(), seed=0, fuzz=1.0, ticks=True, wrap=0.15):
    """Paint yarn strands. Each strand: dict(P=(n, 3) centre line, N=(n, 3) outward normals,
    ok=(n,) which points are in view)."""
    cv, cam, lt, rng, px = sc.cv, sc.cam, sc.lt, sc.rng, sc.px
    Lw = lt.light.L
    bed = Batch(cv, Brush(radius=1.0, hardness=0.2, thick=1.3, pickup=0.4))
    body = Batch(cv, Brush(radius=1.0, hardness=0.25, thick=1.2, pickup=0.4))
    light = Batch(cv, Brush(radius=1.0, hardness=0.15, thick=0.8, pickup=0.5))
    tick = Batch(cv, Brush(radius=1.0, hardness=0.2, thick=0.5, pickup=0.4))
    hair = Batch(cv, Brush(radius=px(0.5), hardness=0.4, thick=0.45, pickup=0.2, press_radius=0.7))
    bloom = Batch(cv, Brush(kind="bristle", radius=1.0, nbristle=9, bristle_r=px(0.7), jitter=0.2, thick=0.22,
                            pickup=0.1, tooth=0.5, load=8))
    for s in strands:
        P, N, ok = s["P"], s["N"], s["ok"]
        xy = cam.project(P)
        ppc = cam.px_per_cm(P)
        col = lt.shade(wool_albedo(P, seed), P, N, skip=skip, wrap=wrap)
        # which way, on the canvas, the window lies from each point of the strand
        to_light = cam.project(P + 0.5 * Lw) - xy
        to_light /= np.maximum(np.linalg.norm(to_light, axis=1, keepdims=True), 1e-9)
        # runs of visible points, cut into short lengths so the colour can change along the strand
        idx = np.nonzero(ok)[0]
        if len(idx) < 3:
            continue
        runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        for run in runs:
            if len(run) < 3:
                continue
            step = max(3, int(round(px(16) / max(1e-6, np.linalg.norm(xy[run[1]] - xy[run[0]])))))
            for a in range(0, len(run) - 1, step):
                seg = run[a:a + step + 1]
                if len(seg) < 2:
                    continue
                m = seg[len(seg) // 2]
                r = STRAND * ppc[m] * (0.9 + 0.2 * rng.random())       # hand-spun: never quite even
                c = col[m] * (0.86 + 0.28 * rng.random())
                tang = xy[seg[-1]] - xy[seg[0]]
                tang /= max(np.linalg.norm(tang), 1e-9)
                nrm = np.array([-tang[1], tang[0]])
                side = nrm * np.sign(nrm @ to_light[m] + 1e-9)
                bed.add(xy[seg], np.clip(c * 0.55, 0, 1), 1.0, r * 1.05)
                body.add(xy[seg] + side * r * 0.10, np.clip(c, 0, 1), 1.0, r * 0.85)
                light.add(xy[seg] + side * r * 0.40, np.clip(c * 1.28 + 0.03, 0, 1), np.linspace(0.6, 1.0, len(seg)), r * 0.45)
                seglen = np.linalg.norm(xy[seg[-1]] - xy[seg[0]])
                if ticks:
                    # the twist of the plies: soft slanting darks across the strand, a little irregular
                    t = rng.random() * 0.3
                    while t < 1.0:
                        q = xy[seg[0]] + (xy[seg[-1]] - xy[seg[0]]) * t
                        tick.add(np.array([q - nrm * r * 0.7 - tang * r * 0.5, q + nrm * r * 0.7 + tang * r * 0.5]),
                                 np.clip(c * 0.66, 0, 1), [0.4, 1.0], r * 0.28)
                        t += max(0.16, px(rng.uniform(5.0, 8.5)) / max(seglen, 1e-6))
                # the bloom of loose fibre lying over the strand
                if rng.random() < 0.7:
                    bloom.add(xy[seg] + side * r * rng.uniform(-0.3, 0.5), np.clip(c * 1.5 + 0.05, 0, 1),
                              np.linspace(0.3, 0.6, len(seg)), r * 0.9)
                # stray fibres
                for _ in range(rng.poisson(0.8 * fuzz)):
                    q = xy[seg[rng.integers(len(seg))]] + nrm * r * rng.uniform(-1.0, 1.0)
                    d = tang * rng.choice([-1, 1]) * rng.uniform(0.6, 1.0) + nrm * rng.normal(0, 0.7)
                    ln = px(rng.uniform(4, 11))
                    bend = nrm * rng.normal(0, 0.25) * ln
                    hair.add(np.array([q, q + d * ln * 0.5 + bend * 0.5, q + d * ln + bend]),
                             np.clip(c * rng.uniform(1.1, 1.6) + 0.03, 0, 1), [1.0, 0.7, 0.1])
    bed.flush(order=None)
    cv.dry_now()
    body.flush(order=None)
    light.flush(order=None)
    cv.dry_now()
    tick.flush()
    cv.dry_now()
    bloom.flush()
    hair.flush()
    cv.dry_now()


class Yarn:
    def __init__(self, sc):
        self.sc = sc
        self.c = np.asarray(L.YARN, dtype=np.float64)
        self.R = L.YARN_R
        c = self.c
        self.ball = forms.sphere(c, self.R)
        # two needles thrust through the ball: knob ends up and back, points down on the table in front
        self.needles = [(c + [-6.6, 5.2, -3.4], c + [7.6, -4.0, 6.2]),
                        (c + [2.6, 8.2, -3.4], c + [-4.4, -3.95, 5.2])]
        self.needle_surfs = []
        for a, b in self.needles:
            ln = np.linalg.norm(b - a)
            taper = lambda v, ln=ln: 0.20 * np.clip((1 - v) * ln / 1.8, 0.06, 1.0) ** 0.7
            self.needle_surfs.append((forms.tube(Curve([a, 0.5 * (a + b), b], smooth=False), taper),
                                      forms.sphere(a, 0.36)))

    def occluders(self):
        occ = [Ball("yarn", self.c, self.R)]
        for i, (a, b) in enumerate(self.needles):
            occ.append(Capsule("needle%d" % i, a, b, 0.2))
        return occ

    def ground_ao(self, P):
        d = np.hypot(P[:, 0] - self.c[0], P[:, 2] - self.c[2])
        return 1 - 0.8 * (1 - smoothstep(0.5, 4.6, d))

    def prepare(self):
        sc, cam = self.sc, self.sc.cam
        W, H = sc.W, sc.H
        self.st_ball = Stencil.from_surface(cam, forms.sphere(self.c, self.R + STRAND), (0, TWO_PI),
                                            (-np.pi / 2, np.pi / 2), W, H, grid=120)
        self.st_needles = []
        self.stencil = self.st_ball
        for tube, knob in self.needle_surfs:
            st = (Stencil.from_surface(cam, tube, (0, TWO_PI), (0, 1), W, H, grid=160)
                  | Stencil.from_surface(cam, knob, (0, TWO_PI), (-np.pi / 2, np.pi / 2), W, H, grid=24))
            self.st_needles.append(st)
            self.stencil = self.stencil | st
        return self.stencil

    def _needle(self, i, vr=(0.0, 1.0), knob=True):
        sc, lt = self.sc, self.sc.lt
        tube, ksurf = self.needle_surfs[i]
        a, b = self.needles[i]
        skip = ("needle%d" % i,) + (("yarn",) if vr[0] > 0 else ())

        def col(P, N, u, v):
            alb = NEEDLE * (0.85 + 0.3 * fbm3(P * [3, 3, 3], 1.0, 2, 97 + i))[:, None]
            return lt.shade(alb, P, N, skip=skip, spec=0.25, gloss=30)
        st = self.st_needles[i]
        paint_form(sc, tube, (0, TWO_PI), vr, col, st, r1=2.2, r2=1.2, along="v", grid=120, blend=1.5, r_detail=0.9)
        if knob:
            paint_form(sc, ksurf, (0, TWO_PI), (-np.pi / 2, np.pi / 2), col, st, r1=2.0, r2=1.1, along="u", grid=24,
                       blend=1.5)

    def paint(self):
        sc = self.sc
        cv, cam, lt, rng, px = sc.cv, sc.cam, sc.lt, sc.rng, sc.px
        if not hasattr(self, "st_ball"):
            self.prepare()
        c, R = self.c, self.R

        # needles first, whole: the ball will cover the part that runs through and behind it
        for i in range(2):
            self._needle(i)
        cv.dry_now()

        # the mass of the ball: deep shadow-colour wool, modelled as a sphere
        def bed(P, N, u, v):
            return lt.shade(wool_albedo(P) * 0.30, P, N, skip=("yarn",), wrap=0.3)
        st_bed = Stencil.from_surface(cam, forms.sphere(c, R - 0.12), (0, TWO_PI), (-np.pi / 2, np.pi / 2),
                                      sc.W, sc.H, grid=120)
        paint_form(sc, forms.sphere(c, R - 0.12), (0, TWO_PI), (-np.pi / 2, np.pi / 2), bed, st_bed, r1=7, r2=3,
                   along="u", grid=60, blend=4.0)
        cv.dry_now()

        # the winding: bundles of parallel turns, each bundle lying over the ones before
        view = cam.to_eye(c)
        wr = np.random.default_rng(5)
        bundles = []
        for k in range(20):
            # early bundles lie anyhow; the later ones are chosen to cross the side we see
            a = normalize(wr.normal(0, 1, 3)) if k < 8 else normalize(np.cross(view, wr.normal(0, 1, 3)) + 0.25 * wr.normal(0, 1, 3))
            bundles.append((a, wr.uniform(-0.6, 0.6) * R * (0.9 if k < 14 else 0.55), int(wr.integers(3, 8))))
        for k, (axis, d0, turns) in enumerate(bundles):
            e1, e2 = forms.frame(axis)
            strands = []
            for j in range(turns):
                d = d0 + (j - turns / 2) * 2 * STRAND * 0.97
                if abs(d) > 0.93 * R:
                    continue
                rad = np.sqrt(R * R - d * d)
                t = np.linspace(0, TWO_PI, 260)
                wob = 0.05 * np.sin(t * 2 + 1.7 * j + k) + 0.03 * np.sin(t * 5 + j)   # turns are never quite parallel
                P = c + (d + wob)[:, None] * axis + rad * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)
                N = normalize(P - c)
                P = c + N * (R + STRAND * 0.5)
                ok = np.sum(N * cam.to_eye(P), axis=1) > 0.06
                strands.append(dict(P=P, N=N, ok=ok))
            if not strands:
                continue
            # the shadow a bundle throws on what lies under it
            if k > 0:
                sh = Batch(cv, Brush(radius=1.0, hardness=0.15, thick=0.35, pickup=0.0))
                for s in (strands[0], strands[-1]):
                    xy = cam.project(s["P"] - 0.05 * lt.light.L * 3)
                    idx = np.nonzero(s["ok"])[0]
                    for run in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
                        if len(run) > 3:
                            sh.add(xy[run], None, 1.0, px(STRAND * 60), paint=Paint.glaze("#101820", 1.4))
                sh.flush(order=None)
                cv.dry_now()
            paint_strands(sc, strands, skip=("yarn",), seed=k, fuzz=0.6 if k < 14 else 1.2)

        # the outline of a wool ball is never clean: fibres stand off it all round
        hair = Batch(cv, Brush(radius=px(0.5), hardness=0.4, thick=0.4, pickup=0.2, press_radius=0.7))
        cxy = cam.project(c)
        for _ in range(420):
            # a point on the limb of the sphere as seen from the eye
            e1, e2 = forms.frame(view)
            ang = rng.uniform(0, TWO_PI)
            n = np.cos(ang) * e1 + np.sin(ang) * e2
            P = c + n * (R + STRAND)
            q = cam.project(P)
            out = (q - cxy) / max(np.linalg.norm(q - cxy), 1e-9)
            colr = lt.shade(wool_albedo(P[None]) * 1.3, P[None], n[None], skip=("yarn",), wrap=0.5)[0]
            ln = px(rng.uniform(3, 11))
            d = out * rng.uniform(0.3, 1.0) + np.array([-out[1], out[0]]) * rng.normal(0, 0.8)
            hair.add(np.array([q - out * px(1.5), q + d * ln * 0.5, q + d * ln + rng.normal(0, px(1.2), 2)]),
                     np.clip(colr, 0, 1), [1.0, 0.6, 0.1])
        hair.flush()
        cv.dry_now()

        # the ends of the needles that come out toward us
        for i, (a, b) in enumerate(self.needles):
            dvec = b - a
            ln = np.linalg.norm(dvec)
            u = dvec / ln
            oc = a - c
            bq = oc @ u
            disc = bq * bq - (oc @ oc - (R + STRAND) ** 2)
            t_exit = (-bq + np.sqrt(max(disc, 0.0))) / ln
            self._needle(i, vr=(min(t_exit + 0.004, 0.999), 1.0), knob=False)
        cv.dry_now()


def thread_path():
    """Where the loose strand runs: from under the ball, along the front of the table, dipping over
    the edge, round in front of the orange and back to the blade. Returns (main run, cut-off end)."""
    s = STRAND
    main = np.array([
        [-10.6, 2.2, 4.6], [-10.2, 0.6, 5.4], [-9.9, s, 6.4], [-9.2, s, 8.6], [-8.4, s, 11.4], [-6.8, s, 14.0],
        [-4.4, s, 15.6], [-2.2, s, 16.5], [-1.0, -0.15, 17.0 + s], [0.2, -1.5, 17.0 + s], [1.6, -1.9, 17.0 + s],
        [2.9, -1.2, 17.0 + s], [3.7, -0.05, 17.0 + s], [4.6, s, 16.2], [6.4, s, 14.9], [8.6, s, 15.9],
        [11.6, s, 16.4], [14.6, s, 16.0], [17.2, s, 14.2], [19.0, s, 11.2], [19.7, s, 8.0], [19.4, s, 5.6],
        [19.05, s, 4.25]])
    cut = np.array([[19.0, s, 3.55], [19.3, s, 2.6], [20.2, s, 1.7], [21.5, s, 1.5], [22.6, s, 2.0]])
    return catmull(main, 0.12), catmull(cut, 0.12)


class Thread:
    """The strand on the table. It is painted last, over everything it lies across."""

    def __init__(self, sc):
        self.sc = sc
        self.main, self.cut = thread_path()

    def occluders(self):
        occ = []
        for path in (self.main, self.cut):
            pts = resample(path, 0.7)
            for a, b in zip(pts[:-1], pts[1:]):
                occ.append(Capsule("thread", a, b, STRAND))
        return occ

    def ground_ao(self, P):
        return np.ones(len(P))

    def prepare(self):
        sc, cam = self.sc, self.sc.cam
        polys = []
        for path in (self.main, self.cut):
            xy = cam.project(path)
            r = STRAND * cam.px_per_cm(path)
            d = np.gradient(xy, axis=0)
            n = np.stack([-d[:, 1], d[:, 0]], 1)
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            polys.append(np.concatenate([xy + n * r[:, None], (xy - n * r[:, None])[::-1]]))
        self.stencil = Stencil.from_polys(polys, sc.W, sc.H)
        self.reserve = Stencil.empty(sc.W, sc.H)       # too slight to leave a gap for: painted over the table
        return self.stencil

    def paint(self):
        sc = self.sc
        cv, cam, rng, px = sc.cv, sc.cam, sc.rng, sc.px
        strands = []
        for path in (self.main, self.cut):
            on_front = path[:, 2] > L.TABLE_FRONT
            N = np.where(on_front[:, None], np.array([0.0, 0.25, 0.97]), np.array([0.0, 1.0, 0.1]))
            strands.append(dict(P=path, N=normalize(N), ok=np.ones(len(path), bool)))
        paint_strands(sc, strands, skip=("thread",), seed=40, fuzz=1.8, wrap=0.3)
        # where the blade came down the yarn is cut: both ends fray open
        hair = Batch(cv, Brush(radius=px(0.55), hardness=0.4, thick=0.6, pickup=0.2, press_radius=0.7))
        for end, back in ((self.main[-1], self.main[-8]), (self.cut[0], self.cut[7]), (self.cut[-1], self.cut[-8])):
            q = cam.project(end)
            d = q - cam.project(back)
            d /= max(np.linalg.norm(d), 1e-9)
            n = np.array([-d[1], d[0]])
            colr = sc.lt.shade(wool_albedo(end[None]) * 1.2, end[None], np.array([[0.0, 1.0, 0.0]]), skip=("thread",), wrap=0.4)[0]
            for _ in range(16):
                spread = rng.normal(0, 0.55)
                ln = px(rng.uniform(5, 14))
                hair.add(np.array([q - d * px(2), q + (d + n * spread) * ln * 0.5, q + (d + n * spread * 1.6) * ln]),
                         np.clip(colr * rng.uniform(0.7, 1.4), 0, 1), [1.0, 0.7, 0.15])
        hair.flush()
        cv.dry_now()
