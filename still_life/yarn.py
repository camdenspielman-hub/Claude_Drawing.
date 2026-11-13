"""The ball of yarn with its needles, and the strand dragged from it across the table.

Hand-spun, woad-dyed wool, wound by hand into a lumpy ball: bundles of turns lying over one
another at every angle, each strand matte, lit along its crest and dark beneath, with a soft
pale halo of fibre on the side toward the window. The needles are what was used then: thin
polished steel wires pointed at both ends, each no more than a bright line beside a dark one.
"""
import numpy as np

from atelier import forms
from atelier.brushwork import Batch
from atelier.canvas import Brush, Paint
from atelier.forms import Curve, fbm3, noise3
from atelier.geom import Ball, Capsule, Glow, catmull, normalize, resample, smoothstep
from atelier.shading import lin
from atelier.solid import Solid
from atelier.stencil import Stencil
from atelier.studio import Form
from . import scene as L

TWO_PI = 2 * np.pi
UP = np.array([0.0, 1.0, 0.0])
STRAND = 0.115                      # radius of one strand of yarn, cm
JAWF = -1.31                        # the level the skull's lower jaw rests at, in the skull's own frame
WOOL = lin("#4f7896")
WOOL_DEEP = lin("#243f55")
WOOL_PALE = lin("#a9c3d6")
SQUASH = np.array([1.04, 0.92, 0.98])       # a wound ball is never round, and it flattens where it sits


def wool_albedo(P, seed=0):
    """Woad blue, unevenly taken up by the fleece: greyer here, deeper there."""
    n = fbm3(P * 0.9 + seed * 3.1, 1.0, 3, 91)[:, None]
    h = noise3(P * 3.0 + seed, 1.0, 93)[:, None]
    base = WOOL * (1 - 0.45 * smoothstep(0.35, 0.75, n)) + WOOL_DEEP * 0.45 * smoothstep(0.35, 0.75, n)
    grey = np.mean(base, axis=1, keepdims=True)
    return (base * 0.86 + grey * 0.14) * (0.85 + 0.3 * h)


def paint_strands(sc, strands, lt, skip=(), seed=0, fuzz=1.0, wrap=0.2, lit_dash=True):
    """Paint yarn strands. Each strand: dict(P=(n, 3) centre line, N=(n, 3) outward normals,
    ok=(n,) which points are in view). Each is laid as a dark bed, the body of the wool, and a
    lighter dash along its crest on the lit side only; then the slant of its twist, and loose fibre."""
    cv, cam, rng, px = sc.cv, sc.cam, sc.rng, sc.px
    Lw = lt.light.L
    bed = Batch(cv, Brush(radius=1.0, hardness=0.2, thick=1.3, pickup=0.4))
    body = Batch(cv, Brush(radius=1.0, hardness=0.25, thick=1.2, pickup=0.4))
    crest = Batch(cv, Brush(radius=1.0, hardness=0.15, thick=0.9, pickup=0.4))
    tick = Batch(cv, Brush(radius=1.0, hardness=0.2, thick=0.45, pickup=0.4))
    hair = Batch(cv, Brush(radius=px(0.5), hardness=0.4, thick=0.4, pickup=0.2, press_radius=0.7))
    for s in strands:
        P, N, ok = s["P"], s["N"], s["ok"]
        xy = cam.project(P)
        ppc = cam.px_per_cm(P)
        col = lt.shade(wool_albedo(P, seed), P, N, skip=skip, wrap=wrap, crowd=0.7)
        facing_light = np.clip(N @ Lw, 0, 1) * lt.light.lit(P, skip)
        to_light = cam.project(P + 0.5 * Lw) - xy
        to_light /= np.maximum(np.linalg.norm(to_light, axis=1, keepdims=True), 1e-9)
        idx = np.nonzero(ok)[0]
        if len(idx) < 3:
            continue
        for run in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
            if len(run) < 3:
                continue
            step = max(3, int(round(px(15) / max(1e-6, np.linalg.norm(xy[run[1]] - xy[run[0]])))))
            for a in range(0, len(run) - 1, step):
                seg = run[a:a + step + 1]
                if len(seg) < 2:
                    continue
                m = seg[len(seg) // 2]
                r = STRAND * ppc[m] * (0.82 + 0.36 * rng.random())        # hand-spun: thick and thin
                c = col[m] * (0.86 + 0.28 * rng.random())
                tang = xy[seg[-1]] - xy[seg[0]]
                seglen = max(np.linalg.norm(tang), 1e-9)
                tang = tang / seglen
                nrm = np.array([-tang[1], tang[0]])
                side = nrm * np.sign(nrm @ to_light[m] + 1e-9)
                bed.add(xy[seg], np.clip(c * 0.40, 0, 1), 1.0, r * 1.10)
                body.add(xy[seg] + side * r * 0.24, np.clip(c, 0, 1), 1.0, r * 0.78)
                if lit_dash and facing_light[m] > 0.25 and rng.random() < 0.85:
                    k = min(len(seg), max(2, int(len(seg) * rng.uniform(0.5, 1.0))))
                    crest.add(xy[seg[:k]] + side * r * 0.42, np.clip(c * (1.18 + 0.30 * facing_light[m]) + 0.03, 0, 1),
                              np.linspace(0.5, 1.0, k), r * 0.42)
                t = rng.random() * 0.3
                while t < 1.0:              # the plies wind round: soft slanting darks
                    q = xy[seg[0]] + (xy[seg[-1]] - xy[seg[0]]) * t
                    tick.add(np.array([q - nrm * r * 0.75 - tang * r * 0.9, q + nrm * r * 0.75 + tang * r * 0.9]),
                             np.clip(c * 0.80, 0, 1), [0.5, 1.0], r * 0.36)
                    t += max(0.16, px(rng.uniform(4.5, 7.5)) / seglen)
                for _ in range(rng.poisson(0.7 * fuzz)):
                    q = xy[seg[rng.integers(len(seg))]] + nrm * r * rng.uniform(-1.0, 1.0)
                    d = tang * rng.choice([-1, 1]) * rng.uniform(0.6, 1.0) + nrm * rng.normal(0, 0.7)
                    ln = px(rng.uniform(4, 11))
                    bend = nrm * rng.normal(0, 0.25) * ln
                    hair.add(np.array([q, q + d * ln * 0.5 + bend * 0.5, q + d * ln + bend]),
                             np.clip(c * rng.uniform(1.05, 1.45) + 0.02, 0, 1), [1.0, 0.7, 0.1],
                             tscale=0.4 + 0.6 * facing_light[m])
    bed.flush(order=None)
    cv.dry_now()
    body.flush(order=None)
    crest.flush(order=None)
    cv.dry_now()
    tick.flush()
    hair.flush()
    cv.dry_now()


class Yarn:
    """The ball: a wound solid. Belts of parallel turns lie over one another in layers, so the
    outline is lumpy, every strand has a lit crest and a dark groove beside it, and the upper
    belts throw small shadows on the ones beneath."""
    softness = (0.55, 0.0)
    reach = 10.0

    def __init__(self, sc):
        self.sc = sc
        self.lt = None
        R = L.YARN_R
        self.R = R
        self.c = np.array([L.YARN[0], R * SQUASH[1] * 0.985, L.YARN[2]])
        c = self.c
        self.core = R - 0.60
        self.pitch = 2 * STRAND / self.core
        wr = np.random.default_rng(11)
        view = normalize(np.asarray(L.EYE) - c)
        self.belts = []
        n = 38
        for i in range(n):
            axis = normalize(wr.normal(0, 1, 3))
            if i >= n - 9:      # the last few turns: narrow belts, crossing the side we see
                axis = normalize(np.cross(view, wr.normal(0, 1, 3)) + 0.35 * wr.normal(0, 1, 3))
            turns = int(wr.integers(5, 13)) if i < n - 9 else int(wr.integers(2, 7))
            off = wr.uniform(-0.55, 0.55) * (1.0 if i < n - 9 else 0.5)
            self.belts.append((axis, off, turns * STRAND / self.core, 0.016 * i))
        self.solid = Solid(np.eye(3), c, 1.0, centre=(0, 0, 0), radius=R + 0.8, rough=(0.07, 0.5), seed=3)
        self.solid.wound_ball([0, 0, 0], SQUASH, self.core, STRAND, self.belts)
        # two needles pushed through the upper part of the ball, crossing; neither touches the table
        self.needles = [(c + [-7.5, 6.0, -2.0], c + [6.5, -1.8, 2.6]),
                        (c + [-4.5, -0.9, 6.5], c + [4.0, 6.8, -4.5])]
        seen = self.solid.seen_from(sc.cam)
        self.seen = seen
        part = seen.outline(sc.W, sc.H)

        def flow(P, N, which):
            return self.strand_at(P)["along"]
        self.ball_form = Form("yarn_ball", seen, (0, sc.W), (0, sc.H), self._colour, owner="yarn", size=4.2,
                              stencil=Stencil(part >= 0), flow=flow, along="u", tol=0.03, broken=0.05, thick=(0.45, 1.3))
        self.needle_forms = []
        for i, (a, b) in enumerate(self.needles):
            ln = np.linalg.norm(b - a)

            def radius(v, ln=ln):       # a wire tapering to a point at each end
                return 0.12 * np.clip(np.minimum(v, 1 - v) * ln / 1.4, 0.05, 1.0) ** 0.6
            tube = forms.tube(Curve([a, 0.5 * (a + b), b], smooth=False), radius)

            def steel(P, N, u, v, i=i):
                lt = self.lt or sc.lt
                return lt.mirror(np.array([0.66, 0.67, 0.68]), P, N, skip=("needle%d" % i,), rough=0.06, taps=3,
                                 dirt=(lin("#3a3836"), np.full(len(P), 0.22)), rng=np.random.default_rng(len(P) + i))

            def shows(P, N, u, v):      # not where it runs through the wool, nor behind it
                shape = np.shape(P)[:-1]
                Q = np.reshape(P, (-1, 3))
                return ((self.solid.distance(Q) > 0.0) & ~self.hides(Q)).reshape(shape)
            f = Form("needle%d" % i, tube, (0, TWO_PI), (0, 1), steel, owner="needles", size=1.3, grid=140, along="v", dead=False,
                     tol=0.02, broken=0.0, outline_grid=260, vis=shows)
            self.needle_forms.append(f)
        self.forms = [self.ball_form] + self.needle_forms

    @property
    def centre(self):
        return self.c

    def hides(self, P):
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        eye = self.sc.cam.eye
        d = P - eye
        dist = np.linalg.norm(d, axis=1)
        t, _ = self.solid.trace(np.broadcast_to(eye, P.shape), d / dist[:, None])
        return ~np.isnan(t) & (t < dist - 0.05)

    def strand_at(self, P):
        """Which turn of which belt lies uppermost at points P of the ball: the way the strand runs
        there, how far across it the point is (-0.5 .. 0.5), and a number for the turn."""
        p = (np.asarray(P, dtype=np.float64) - self.c) / SQUASH
        d = normalize(p)
        n = len(d)
        best = np.zeros(n)
        belt = np.full(n, -1)
        across = np.zeros(n)
        turn = np.zeros(n)
        axis = np.tile(UP, (n, 1))
        for b, (ax, off, half, lift) in enumerate(self.belts):
            s = d @ ax - off
            e = s / self.pitch
            j = np.floor(e + 0.5)
            e = e - j
            prof = STRAND * (0.40 + 0.60 * np.cos(np.pi * e))
            edge = np.minimum(1.0, (half - np.abs(s)) / (0.5 * self.pitch))
            hi = lift + prof * (0.55 + 0.45 * edge)
            up = (np.abs(s) <= half) & (hi > best)
            best[up], belt[up], across[up], turn[up] = hi[up], b, e[up], j[up] + 100 * b
            axis[up] = ax
        along = np.cross(axis, d)
        along = normalize(np.where(np.linalg.norm(along, axis=1, keepdims=True) < 1e-6, [1.0, 0, 0], along))
        e1 = normalize(np.cross(axis, np.where(np.abs(axis[:, :1]) < 0.9, [1.0, 0, 0], [0, 1.0, 0])))
        e2 = np.cross(axis, e1)
        round_ = np.arctan2(np.sum(d * e2, axis=1), np.sum(d * e1, axis=1))
        return dict(belt=belt, across=across, turn=turn, along=along, round=round_)

    def _colour(self, P, N, u, v):
        lt = self.lt or self.sc.lt
        sol = self.solid
        st = self.strand_at(P)
        alb = wool_albedo(P)
        # no two lengths of hand-spun, hand-dyed yarn are quite one colour
        h = np.modf(np.abs(np.sin(st["turn"] * 12.9898 + 4.1)) * 43758.5453)[0][:, None]
        alb = alb * (0.84 + 0.32 * h)
        # two plies twisted together: soft slanting darks along every strand
        ply = 0.5 + 0.5 * np.sin(TWO_PI * (st["round"] * self.core / 0.40 + st["across"]))
        alb = alb * (0.88 + 0.16 * ply)[:, None]
        alb = np.where((st["belt"] < 0)[:, None], alb * 0.4, alb)
        groove = sol.openness(P, N, reach=0.22)
        nest = sol.openness(P, N, reach=1.1)
        lit = lt.light.lit(P, ("yarn",)) * sol.sees(P, N, lt.light.L, reach=9.0, soft=5.0)
        # loose fibre standing off the wool catches the window where the surface turns away from us
        V = self.sc.cam.to_eye(P)
        graze = (1 - np.clip(np.sum(N * V, axis=1), 0, 1)) ** 3
        fuzz = (graze * (0.02 + 0.98 * lit * np.clip(N @ lt.light.L + 0.25, 0, 1)))[:, None] * WOOL_PALE * 0.16
        sits = 0.30 + 0.70 * smoothstep(0.05, 0.9, P[:, 1])              # the dark crease where it rests on the table
        shut = np.clip(groove, 0, 1) * (0.50 + 0.50 * np.clip(nest, 0, 1)) * sits
        # wool in shadow is still blue: the dim room fills it, and near the table a little warmth comes up
        fill = alb * 0.085 * shut[:, None] + alb.mean(axis=1, keepdims=True) * np.array([0.05, 0.03, 0.015]) * (smoothstep(2.6, 0.3, P[:, 1]) * sits)[:, None]
        return lt.shade(alb, P, N, skip=("yarn",), ao=shut, lit=lit, wrap=0.24, crowd=0.8, emit=fuzz + fill)

    def occluders(self):
        occ = [Ball("yarn", self.c, 1.0, scale=tuple(self.R * SQUASH))]
        for i, (a, b) in enumerate(self.needles):
            occ.append(Capsule("needle%d" % i, a, b, 0.12))
        return occ

    def glows(self):
        return [Glow("yarn", self.c, self.R * 0.95, WOOL)]

    def groups(self):
        return [(self.sc.cam.depth(self.c + [0, -self.c[1], self.R * 0.6]), self.forms)]

    def register(self, studio):
        studio.softness_of["yarn"] = self.softness
        studio.softness_of["needles"] = (0.03, 0.0)
        studio.hook("detail", self.crests)
        studio.hook("finish", self.fibre)

    def crests(self, studio):
        """Fat paint along the crest of the strands that face the window."""
        sc = self.sc
        cv, cam, rng, px = sc.cv, sc.cam, sc.rng, sc.px
        lt = self.lt or sc.lt
        region = self.ball_form.region
        ys, xs = np.nonzero(region.mask)
        if len(xs) == 0:
            return
        n = int(region.area() / px(1.0) ** 2 / 190.0)
        pick = rng.integers(0, len(xs), n)
        q = np.stack([xs[pick] + rng.random(n), ys[pick] + rng.random(n)], 1)
        P, N = self.seen(q[:, 0], q[:, 1])
        ok = np.any(N != 0, axis=1)
        q, P, N = q[ok], P[ok], N[ok]
        st = self.strand_at(P)
        lit = lt.light.lit(P, ("yarn",)) * self.solid.sees(P, N, lt.light.L, reach=9.0, soft=5.0) * np.clip(N @ lt.light.L, 0, 1)
        keep = (np.abs(st["across"]) < 0.22) & (st["belt"] >= 0) & (lit > 0.30)
        q, P, N, lit, along = q[keep], P[keep], N[keep], lit[keep], st["along"][keep]
        if len(q) == 0:
            return
        d2 = cam.project(P + along * 0.05) - cam.project(P)
        d2 /= np.maximum(np.linalg.norm(d2, axis=1, keepdims=True), 1e-9)
        cols = lt.shade(wool_albedo(P) * 1.28, P, N, skip=("yarn",), lit=lit / np.maximum(np.clip(N @ lt.light.L, 1e-3, 1), 1e-3), wrap=0.24)
        dash = Batch(cv, Brush(radius=px(1.3), hardness=0.35, thick=1.5, pickup=0.25, press_radius=0.6))
        for i in range(len(q)):
            ln = px(rng.uniform(7, 16))
            pts = np.array([q[i] - d2[i] * ln / 2, q[i], q[i] + d2[i] * ln / 2])
            if region.inside(pts, px(1.0)).all():
                dash.add(pts, np.clip(cols[i] * rng.uniform(0.95, 1.12), 0, 1), [0.3, 1.0, 0.4], tscale=0.5 + 0.7 * lit[i])
        dash.flush()

    def fibre(self, studio):
        """Loose fibre standing off the outline: a pale soft halo toward the window, almost nothing
        on the dark side; and a dark touch where each needle goes into the wool."""
        sc = self.sc
        cv, cam, rng, px = sc.cv, sc.cam, sc.rng, sc.px
        lt = self.lt or sc.lt
        region = self.ball_form.region
        pts = region.edge_points(px(2.2), rng)
        if len(pts) == 0:
            return
        inward = region.inward(pts)
        P, N = self.seen(*(pts + inward * px(2.5)).T)
        ok = np.any(N != 0, axis=1)
        # only where the ball's own outline is the edge (not where something stands in front of it)
        ok &= ~self.ball_form.stencil.inside(pts - inward * px(3.0))
        pts, inward, P, N = pts[ok], inward[ok], P[ok], N[ok]
        lit = lt.light.lit(P, ("yarn",)) * np.clip(N @ lt.light.L + 0.20, 0, 1)
        cols = lt.shade(wool_albedo(P) * 1.4, P, N, skip=("yarn",), wrap=0.35)
        hair = Batch(cv, Brush(radius=px(0.5), hardness=0.3, thick=0.35, pickup=0.2, press_radius=0.7))
        for i in range(len(pts)):
            for _ in range(rng.poisson(0.02 + 1.3 * lit[i] ** 1.5)):
                out = -inward[i]
                tang = np.array([-out[1], out[0]])
                ln = px(rng.uniform(2, 8))
                d = out * rng.uniform(0.25, 1.0) + tang * rng.normal(0, 0.9)
                q = pts[i] + inward[i] * px(rng.uniform(0.5, 3.0))
                hair.add(np.array([q, q + d * ln * 0.5 + tang * rng.normal(0, px(1.0)), q + d * ln + rng.normal(0, px(1.3), 2)]),
                         np.clip(cols[i] * rng.uniform(0.8, 1.25), 0, 1), [1.0, 0.6, 0.1], tscale=0.25 + 0.75 * lit[i])
        hair.flush()
        # each needle's thin shadow on the table, running away from the window
        Lw = lt.light.L
        table = studio.regions.get("table")
        shade = Batch(cv, Brush(radius=1.0, hardness=0.0, thick=0.4, pickup=0.0))
        for a, b in self.needles:
            line = a + (b - a) * np.linspace(0, 1, 160)[:, None]
            Q = line - Lw * (line[:, 1] / Lw[1])[:, None]
            q = cam.project(Q)
            ok = (table.inside(q, px(2.0)) if table is not None else np.ones(len(q), bool)) & (lt.light.lit(Q + [0, 0.01, 0], ("needle",)) > 0.5)
            ppc = cam.px_per_cm(Q)
            idx = np.nonzero(ok)[0]
            for run in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
                for a0 in range(0, len(run) - 1, 5):
                    seg = run[a0:a0 + 6]
                    if len(seg) >= 2:
                        h = float(line[seg[0], 1])
                        shade.add(q[seg], None, 1.0, ppc[seg[0]] * (0.14 + 0.045 * h),
                                  paint=Paint.glaze("#1c1108", float(np.clip(1.1 - 0.10 * h, 0.25, 1.1))))
        shade.flush(order=None)
        cv.dry_now()
        for a, b in self.needles:
            ts = np.linspace(0, 1, 400)
            line = a + (b - a) * ts[:, None]
            d = self.solid.distance(line)
            for k in np.nonzero((d[:-1] > 0) != (d[1:] > 0))[0]:
                Pn = line[k]
                if self.hides(Pn[None])[0]:
                    continue
                c2 = cam.project(Pn[None])[0]
                inward2 = cam.project((line[k + 1] if d[k + 1] < 0 else line[k - 1])[None])[0] - c2
                inward2 /= max(np.linalg.norm(inward2), 1e-9)
                cv.stroke(np.array([c2 - inward2 * px(1.0), c2 + inward2 * px(5.0)]), Paint.opaque("#0c1218"),
                          Brush(radius=px(2.2), hardness=0.2, thick=0.7, pickup=0.2), [1.0, 0.3])
        cv.dry_now()


def strand_course(sc):
    """Where the dragged strand runs. From the ball it wanders to the skull and into its mouth beside
    the fang; it comes out under the back of the skull, runs on to the blade, and there it is cut.
    The cut end beyond the blade trails to the table's edge and hangs over it.
    Returns a list of (points, hidden) pieces; hidden pieces are inside the skull."""
    s = STRAND
    sk = sc.objs.get("skull")
    y = s
    if sk is not None:
        # up the outside of the lower jaw just behind the fangs, into the gap between fang and cheek teeth
        approach = sk.world([9.6, JAWF + 0.12, 4.3])
        lead = sk.world([9.2, JAWF + 0.14, 2.35])
        over_jaw = sk.world([9.12, -0.42, 1.03])
        mouth_in = sk.world([9.08, -0.08, 0.92])
        inside = sk.world([8.5, -0.10, 0.4])
        out_back = sk.world([1.5, JAWF + 0.12, 0.8])
        for q in (approach, lead, out_back):
            q[1] = y
    else:
        mouth_in = np.array([0.5, 1.0, 12.0]); inside = np.array([1.5, 1.0, 11.0]); out_back = np.array([8.0, s, 7.0])
        approach = np.array([-1.5, s, 14.0]); lead = np.array([-0.3, s, 13.0]); over_jaw = np.array([0.1, 0.6, 12.4])
    ball = np.array([L.YARN[0], 0.0, L.YARN[2]])
    # slack wool does not lie in waves: it runs, kinks back on itself, and throws a flat loop
    a = np.array([
        ball + [3.2, 1.6, 2.6], ball + [3.9, 0.45, 3.4], ball + [4.6, y, 4.4], ball + [6.2, y, 5.2], ball + [6.7, y, 6.9],
        ball + [5.7, y, 8.2], ball + [6.5, y, 9.7], ball + [8.7, y, 9.9], ball + [9.9, y, 8.7], ball + [9.3, y, 7.3],
        ball + [8.1, y, 7.6], ball + [8.5, y, 9.0], ball + [10.6, y, 9.5], ball + [12.3, y, 9.9],
        0.5 * (ball + [12.3, y, 9.9] + approach), approach, lead, over_jaw, mouth_in])
    b = np.array([mouth_in, inside, out_back])
    blade = np.asarray(L.BLADE, dtype=np.float64)
    # from under the back of the skull on to the blade, where it was cut: the two ends have sprung apart
    c = np.array([out_back, out_back + [1.2, 0.0, 0.7], [out_back[0] + 2.4, y, out_back[2] + 2.3],
                  [11.6, y, 8.9], [13.9, y, 8.2], [15.6, y, 8.9], blade + [-1.7, y, 2.1], blade + [-0.95, y, 1.25],
                  blade + [-0.75, y, 0.55]])
    c[1, 1] = y * 2.2
    d = np.array([blade + [0.70, y, -0.45], blade + [1.25, y, 0.05], blade + [1.7, y, 1.6], blade + [1.25, y, 4.6],
                  blade + [2.1, y, 7.4], blade + [1.7, y, 9.6], blade + [2.35, y, 11.0], blade + [2.45, y * 0.6, 11.4 + s],
                  blade + [2.5, -0.9, 11.55 + s], blade + [2.42, -2.4, 11.6 + s]])
    return [(catmull(a, 0.1), False), (catmull(b, 0.1), True), (catmull(c, 0.1), False), (catmull(d, 0.1), False)]


class Strand:
    """The strand on the table. Painted late, where it lies over things; left out where things lie over it."""
    softness = (0.35, 0.0)
    reach = 40.0

    def __init__(self, sc):
        self.sc = sc
        self.lt = None
        self.forms = []
        self._pieces = None

    def pieces(self):
        if self._pieces is None:
            self._pieces = strand_course(self.sc)
            self._lift()
        return self._pieces

    def _lift(self):
        """A strand lying across coins rides up over them."""
        coins = self.sc.objs.get("coins")
        if coins is None:
            return
        for pts, hidden in self._pieces:
            if hidden:
                continue
            base = pts[:, 1].copy()
            for k in coins.coins:
                d = np.hypot(pts[:, 0] - k["c"][0], pts[:, 2] - k["c"][2])
                top = k["c"][1] + k["t"] / 2 + STRAND
                over = smoothstep(k["R"] * 1.25, k["R"] * 0.85, d)
                pts[:, 1] = np.maximum(pts[:, 1], base * (1 - over) + np.maximum(base, top) * over)

    @property
    def centre(self):
        return np.array([4.0, 0.2, 8.0])

    def occluders(self):
        return []           # its thin shadow is drawn beside it by hand (see lay)

    def glows(self):
        return []

    def groups(self):
        return []

    def register(self, studio):
        studio.hook("detail", self.lay)

    def lay(self, studio):
        sc = self.sc
        cv, cam, rng, px = sc.cv, sc.cam, sc.rng, sc.px
        lt = sc.lt
        strands = []
        ends = []
        hide_behind = {"dagger": L.BLADE[2] + 0.05, "orange": None, "hourglass": None}
        for pts, hidden in self.pieces():
            if hidden:
                continue
            on_front = pts[:, 2] > L.TABLE_FRONT
            N = normalize(np.where(on_front[:, None], np.array([0.0, 0.25, 0.97]), np.array([0.0, 1.0, 0.1])))
            xy = cam.project(pts)
            ok = np.ones(len(pts), bool)
            # the skull hides the strand wherever the strand is farther from us than the bone in front of it
            sk = sc.objs.get("skull")
            if sk is not None and "skull" in studio.regions:
                ok &= ~sk.hides(pts)
            for owner, zlim in hide_behind.items():
                if owner in studio.regions:
                    inside = studio.regions[owner].inside(xy)
                    ok &= ~(inside & ((pts[:, 2] < zlim) if zlim is not None else (cam.depth(pts) > cam.depth(sc.objs[owner].centre))))
            strands.append(dict(P=pts, N=N, ok=ok))
        # first its shadow: a thin soft dark beside it, on the side away from the window, wider where it rides up
        shade = Batch(cv, Brush(radius=1.0, hardness=0.0, thick=0.85, pickup=0.0))
        Lw = lt.light.L
        for st_ in strands:
            P, ok = st_["P"], st_["ok"]
            if ok.sum() < 3:
                continue
            base = np.where(P[:, 2] > L.TABLE_FRONT, P[:, 1], np.maximum(P[:, 1] - STRAND, 0.0))
            Q = P - Lw * (base / Lw[1])[:, None] * np.array([1.0, 1.0, 1.0])
            Q[:, 1] = P[:, 1] - base
            q = cam.project(Q)
            ppc = cam.px_per_cm(P)
            idx = np.nonzero(ok & (P[:, 2] <= L.TABLE_FRONT))[0]
            for run in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
                step = 6
                for a0 in range(0, len(run) - 1, step):
                    seg = run[a0:a0 + step + 1]
                    if len(seg) >= 2:
                        shade.add(q[seg] + [px(1.6), px(2.2)], None, 1.0, STRAND * ppc[seg[0]] * 1.35,
                                  paint=Paint.glaze("#0e0703", 2.4))
        shade.flush(order=None)
        cv.dry_now()
        paint_strands(sc, strands, lt, skip=("strand",), seed=40, fuzz=1.5, wrap=0.3)
        # the two cut ends at the blade, and the end that hangs over the table's edge: the plies spring apart a little
        pcs = [p for p, h in self.pieces() if not h]
        hair = Batch(cv, Brush(radius=px(0.5), hardness=0.4, thick=0.55, pickup=0.2, press_radius=0.7))
        for end, back, spread_k in ((pcs[1][-1], pcs[1][-9], 0.45), (pcs[2][0], pcs[2][8], 0.45), (pcs[2][-1], pcs[2][-9], 0.8)):
            q = cam.project(end[None])[0]
            d = q - cam.project(back[None])[0]
            d /= max(np.linalg.norm(d), 1e-9)
            n = np.array([-d[1], d[0]])
            colr = lt.shade(wool_albedo(end[None]) * 1.15, end[None], np.array([[0.0, 1.0, 0.0]]), skip=("strand",), wrap=0.4)[0]
            for spread in (-0.75, 0.1, 0.8):
                ln = px(rng.uniform(9, 15))
                hair.add(np.array([q - d * px(3), q + (d + n * spread * 0.5 * spread_k) * ln * 0.5, q + (d + n * spread * spread_k) * ln]),
                         np.clip(colr * rng.uniform(0.85, 1.15), 0, 1), [1.0, 0.8, 0.25], 3.6)
            for _ in range(14):
                spread = rng.normal(0, 0.5) * spread_k * 2
                ln = px(rng.uniform(5, 14))
                hair.add(np.array([q - d * px(2), q + (d + n * spread) * ln * 0.5, q + (d + n * spread * 1.5) * ln]),
                         np.clip(colr * rng.uniform(0.7, 1.4), 0, 1), [1.0, 0.7, 0.15])
        hair.flush()
        cv.dry_now()


STUDIO_READY = True
