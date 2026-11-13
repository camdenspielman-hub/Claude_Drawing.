"""The blade driven into the table: a small dagger in the manner of the Saxon left-hand daggers of
about 1600, though a good deal shorter than those (it is some 27 cm overall).

A flattened-diamond blade with a short flat base stamped with a maker's mark; a guard of two
rope-twisted iron bars drooping toward the blade, with a twisted ring standing out in front; a
grip bound in fine wire between two braided collars; a pear-shaped, fluted pommel. Nothing on
it is true any more: the ridge wanders, one arm of the guard is bent, the steel is satin grey
with old scratches and pits, and what gilding is left lies in the hollows of the twists.
"""
import numpy as np

from atelier import forms
from atelier.brushwork import Batch
from atelier.canvas import Brush, Paint
from atelier.forms import Curve, Profile, fbm3, noise3
from atelier.geom import Ball, Capsule, Quad, Glow, normalize, smoothstep
from atelier.shading import lin
from atelier.studio import Form
from . import scene as L

TWO_PI = 2 * np.pi
BLADE = 18.0         # blade length, tip to guard
BURIED = 2.4         # how much of it is in the wood
RICASSO = 1.9        # the flat-sided base of the blade
STEEL = np.array([0.56, 0.57, 0.58])
IRON = np.array([0.30, 0.29, 0.28])
GILT = np.array([0.95, 0.68, 0.26])
COPPER = np.array([0.62, 0.36, 0.22])
PATINA = lin("#4a4038")


def half_width(s):
    """Half the blade's width at distance s from the point: a straight, even taper to a needle."""
    s = np.clip(np.asarray(s, dtype=np.float64), 0, None)
    return 1.12 * (0.05 + 0.95 * (s / BLADE) ** 0.85)


class Dagger:
    softness = (0.06, 0.10)
    reach = 16.0

    def __init__(self, sc):
        self.sc = sc
        self.e = np.asarray(L.BLADE, dtype=np.float64)          # where it enters the wood
        self.a = normalize(L.BLADE_AXIS)                        # up the dagger
        a = self.a
        self.bw = normalize(np.cross(a, normalize([-0.30, 0.15, 0.94])))     # across the blade
        self.bn = np.cross(self.bw, a)                                       # the way the flat faces
        self.tip = self.e - BURIED * a
        self.g = self.tip + BLADE * a                           # where blade meets guard
        self.lt = None
        self.forms = []
        self._build()

    @property
    def centre(self):
        return self.tip + (BLADE + 3.0) * self.a

    def occluders(self):
        e, a, bw, g = self.e, self.a, self.bw, self.g
        hw0, hw1 = float(half_width(BURIED)), float(half_width(BLADE))
        occ = [Quad("dagger:blade", [e - hw0 * bw, e + hw0 * bw, g + hw1 * bw, g - hw1 * bw]),
               Capsule("dagger:guard", g + 0.5 * a - 3.6 * bw - 0.7 * a, g + 0.5 * a + 3.6 * bw - 0.5 * a, 0.3),
               Capsule("dagger:grip", g + 1.1 * a, g + 6.6 * a, 0.8),
               Ball("dagger:pommel", g + 7.9 * a, 0.95)]
        return occ

    def glows(self):
        return []

    # ---- forms
    def _build(self):
        sc = self.sc
        e, a, bw, bn, tip, g = self.e, self.a, self.bw, self.bn, self.tip, self.g
        ridge_h = 0.20
        above = lambda P, N, u, v: P[..., 1] > 0.0             # the wood hides the rest

        def add(name, surf, ur, vr, colour, **kw):
            f = Form("dagger_" + name, surf, ur, vr, colour, owner="dagger", **kw)
            self.forms.append(f)
            return f

        def wander(s):                                          # the ridge is not quite down the middle
            return 0.05 * np.sin(s * 0.55 + 0.6) + 0.025 * np.sin(s * 1.7)

        def bevel(sign):
            def surf(u, v):         # u: 0 at the cutting edge, 1 at the ridge; v: distance from the point
                u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
                hw = half_width(v)
                mid = wander(v)
                x = sign * hw * (1 - u) + mid * u
                P = tip + v[..., None] * a + x[..., None] * bw + (ridge_h * u)[..., None] * bn
                N = normalize(np.maximum(hw, 0.05)[..., None] * bn + sign * ridge_h * bw + 0 * P)
                return P, N
            return surf

        def steel(name, sign):
            def col(P, N, u, v):
                lt = self.lt or sc.lt
                # the grinding left the bevels faintly wavy, so what they reflect comes in soft bands
                w1 = noise3(np.stack([v * 0.45, u * 0.5, 0 * v + (2.0 if sign < 0 else 6.0)], 1), 1.0, 131) - 0.5
                w2 = noise3(np.stack([v * 1.1, u * 1.6, 0 * v], 1), 1.0, 133) - 0.5
                Nb = normalize(N + (0.14 * w1 + 0.03 * w2)[:, None] * a + (0.07 * w1 + 0.03 * w2)[:, None] * bw)
                stain = smoothstep(0.50, 0.78, fbm3(P * 1.6, 1.0, 3, 135))                     # grey-brown mottling
                pits = smoothstep(0.78, 0.92, fbm3(P * 9.0, 1.0, 2, 137))
                near_guard = smoothstep(BLADE - 4.5, BLADE, v) * 0.35 + smoothstep(4.5, BURIED, v) * 0.25
                # satin steel is half mirror, half matte grey: the matte part is what makes the bevel
                # turned to the window light and the other one dark
                matte = STEEL * 0.62 * (1 - 0.55 * stain[:, None]) + PATINA * 0.55 * stain[:, None]
                matte = matte * (1 - 0.6 * pits[:, None])
                dirt = np.clip(0.26 + 0.22 * stain + 0.30 * pits + near_guard, 0, 0.92)
                c = lt.mirror(STEEL * 1.25, P, Nb, skip=("dagger:blade",), rough=0.08, taps=5, dirt=(matte, dirt), raw=True)
                return lt.tone(c)
            add(name, bevel(sign), (0, 1), (0.0, BLADE - RICASSO), col, size=3.0, grid=90, along="v", vis=above,
                min_facing=-2.0, tol=0.022, broken=0.02, outline_grid=200)

        # the far bevel first, then the near one (whichever the eye sees more of comes last)
        steel("bevel_r", +1)
        steel("bevel_l", -1)

        # the flat base of the blade, with its two short grooves and the smith's mark
        hw = float(half_width(BLADE))
        ro = tip + (BLADE - RICASSO) * a - hw * bw + ridge_h * 0.95 * bn

        def ricasso_col(P, N, u, v):
            lt = self.lt or sc.lt
            groove = (np.exp(-((u - 0.30) / 0.045) ** 2) + np.exp(-((u - 0.70) / 0.045) ** 2)) * smoothstep(0.12, 0.3, v)
            mark = np.exp(-(((u - 0.5) / 0.07) ** 2 + ((v - 0.52) / 0.10) ** 2))
            Nb = normalize(N + (0.5 * np.gradient(groove) if False else 0.0))
            blotch = smoothstep(0.5, 0.8, fbm3(P * 3.0, 1.0, 3, 139))
            dirt = np.clip(0.34 + 0.50 * groove + 0.4 * mark + 0.25 * blotch, 0, 0.95)
            matte = STEEL * 0.55 * (1 - 0.6 * (groove + 0.6 * blotch).clip(0, 1)[:, None]) + PATINA * 0.6 * (groove + 0.6 * blotch).clip(0, 1)[:, None]
            return lt.mirror(STEEL * 1.15, P, N, skip=("dagger:blade",), rough=0.14, taps=6, dirt=(matte, dirt))
        add("ricasso", forms.plane(ro, 2 * hw * bw, RICASSO * a, flip=np.cross(bw, a) @ bn < 0), (0, 1), (0, 1), ricasso_col,
            size=2.4, grid=24, along="v", flat=(ro, 2 * hw * bw, RICASSO * a), tol=0.02)

        # ---- hilt
        def iron(tint_fn, skip, rough=0.30, twist=None):
            def col(P, N, u, v):
                lt = self.lt or sc.lt
                tint, dirt_amt, Nb = tint_fn(P, N, u, v)
                return lt.mirror(tint, P, Nb, skip=skip, rough=rough, taps=6, dirt=(PATINA * 0.6, dirt_amt),
                                 rng=np.random.default_rng(len(P) + 11))
            return col

        def rope(turns):
            """Rope-twisted iron: dark on the crests where hands have rubbed the gilding away,
            gold still lying in the hollows."""
            def tint_fn(P, N, u, v):
                ph = u + v * turns * TWO_PI
                crest = 0.5 + 0.5 * np.cos(ph * 3.0)
                tint = IRON * crest[:, None] + GILT * 0.75 * (1 - crest[:, None])
                Nb = normalize(N + (0.45 * np.sin(ph * 3.0))[:, None] * self.bw * 0.0 + 0.0)
                bump = 0.26 * np.sin(ph * 3.0) * (0.6 + 0.6 * fbm3(P * 2.2, 1.0, 2, 153))
                tang = normalize(np.cross(N, a))
                Nb = normalize(N + bump[:, None] * tang)
                dirt = np.clip(0.30 + 0.35 * (1 - crest) * 0 + 0.3 * fbm3(P * 5.0, 1.0, 2, 141), 0, 0.9)
                return tint, dirt, Nb
            return tint_fn

        block_c = g + 0.55 * a
        # the two arms: one droops as made, the other has been knocked a little out of true
        for k, (sgn, droop, bend) in enumerate(((-1, 1.00, 0.0), (1, 0.82, 0.07))):
            pts = [block_c + sgn * 1.15 * bw, block_c + sgn * 2.3 * bw - 0.18 * droop * a + bend * 0.3 * bn,
                   block_c + sgn * 3.25 * bw - 0.62 * droop * a + bend * bn, block_c + sgn * 3.85 * bw - 1.20 * droop * a + bend * 1.2 * bn]
            arm = forms.tube(Curve(pts), lambda v: 0.25 + 0.03 * np.sin(v * 9.0))
            add("arm%d" % k, arm, (0, TWO_PI), (0, 1), iron(rope(5.5), ("dagger:guard",)), size=2.0, grid=60, along="v",
                tol=0.02, broken=0.02)
            knob = forms.roughened(forms.sphere(pts[-1] - 0.05 * a, 0.36), 0.03, 2.0, seed=k)
            add("knob%d" % k, knob, (0, TWO_PI), (-np.pi / 2, np.pi / 2),
                iron(lambda P, N, u, v: (IRON * 0.9 + GILT * 0.15, np.clip(0.35 + 0.3 * fbm3(P * 6.0, 1.0, 2, 143), 0, 0.9), N),
                     ("dagger:guard",)), size=1.6, grid=20, along="u")
        # the ring that stands out in front of the guard
        th = np.linspace(-0.15, np.pi + 0.15, 40)
        ring_pts = block_c + 1.25 * np.cos(th)[:, None] * bw + (0.25 + 1.25 * np.sin(th))[:, None] * bn - 0.12 * np.sin(th)[:, None] * a
        add("ring", forms.tube(Curve(ring_pts), lambda v: 0.21 + 0 * v), (0, TWO_PI), (0, 1), iron(rope(7.0), ("dagger:guard",)),
            size=1.8, grid=70, along="v", tol=0.02, broken=0.02)
        # the block the arms and ring grow from: three faces of a small iron box
        bx, by, bz = 1.25, 0.55, 0.50
        for name, origin, eu, ev, n in (("block_f", block_c - bx * bw - by * a + bz * bn, 2 * bx * bw, 2 * by * a, bn),
                                        ("block_t", block_c - bx * bw + by * a - bz * bn, 2 * bx * bw, 2 * bz * bn, a),
                                        ("block_s", block_c - bx * bw - by * a - bz * bn, 2 * bz * bn, 2 * by * a, -bw)):
            add(name, forms.plane(origin, eu, ev, flip=np.cross(eu, ev) @ n < 0), (0, 1), (0, 1),
                iron(lambda P, N, u, v: (IRON, np.clip(0.4 + 0.4 * fbm3(P * 5.0, 1.0, 3, 145), 0, 0.95), N), ("dagger:guard",), rough=0.35),
                size=2.0, grid=12, along="u", flat=(origin, eu, ev), finish=0.7)

        # the grip: fine twisted wire, wound close, between two braided collars
        grip_prof = Profile([(0.0, 0.78), (0.25, 0.86), (0.5, 0.78), (0.7, 0.70), (2.9, 0.98), (5.0, 0.72), (5.2, 0.80), (5.45, 0.88), (5.7, 0.80)])
        grip_base = g + 1.12 * a

        def wire(P, N, u, v):
            collar = (v < 0.6) | (v > 5.1)
            # wound by hand and worn by one: the turns crowd and gape, and some have sunk under grime
            turns = np.where(collar, v * 3.2 + u * 0.5, v * 8.5 + 0.35 * np.sin(v * 2.3 + u) + 0.2 * np.sin(v * 5.1 + 2 * u))
            ph = turns * TWO_PI
            worn = 0.35 + 0.65 * fbm3(P * 1.7, 1.0, 2, 151)
            bump = np.where(collar, 0.45, 0.28) * np.sin(ph) * worn
            Nb = normalize(N + bump[:, None] * a)
            # a fine second twist within each turn
            fine = 0.5 + 0.5 * np.sin(u * 14.0 + ph * 0.5)
            tint = COPPER * (0.70 + 0.35 * fine)[:, None]
            dirt = np.clip(0.42 + 0.30 * (0.5 - 0.5 * np.sin(ph)) + 0.25 * fbm3(P * 4.0, 1.0, 2, 147), 0, 0.95)   # grime between the turns
            return tint, dirt, Nb
        add("grip", forms.revolution(grip_base, a, grip_prof), (0, TWO_PI), (0, 5.7), iron(wire, ("dagger:grip",), rough=0.22),
            size=2.4, grid=120, along="u", tol=0.02, broken=0.02, outline_grid=200)

        # the pommel: a pear, cut with spiral flutes, the tang riveted over a small button on top
        pear = Profile([(0.0, 0.52), (0.25, 0.66), (0.9, 0.92), (1.45, 0.86), (1.95, 0.52), (2.1, 0.34)])
        pom_base = grip_base + 5.72 * a
        e1, e2 = forms.frame(a)

        def pommel(u, v):
            u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
            flute = 1 - 0.12 * (0.5 + 0.5 * np.cos(7 * (u + 0.75 * v)))
            r = pear(v) * flute
            rad = np.cos(u)[..., None] * e1 + np.sin(u)[..., None] * e2
            P = pom_base + v[..., None] * a + r[..., None] * rad
            dr = (pear(v + 1e-3) - pear(v - 1e-3)) / 2e-3
            dflute = 0.12 * 0.5 * 7 * np.sin(7 * (u + 0.75 * v))
            tang = -np.sin(u)[..., None] * e1 + np.cos(u)[..., None] * e2
            N = normalize(rad - dr[..., None] * a - (dflute * 0.5)[..., None] * tang)
            return P, N

        def pommel_tint(P, N, u, v):
            hollow = 0.5 + 0.5 * np.cos(7 * (u + 0.75 * v))
            tint = IRON * (1 - 0.5 * hollow[:, None]) + GILT * 0.6 * 0.5 * hollow[:, None]
            return tint, np.clip(0.30 + 0.30 * fbm3(P * 5.0, 1.0, 2, 149), 0, 0.9), N
        add("pommel", pommel, (0, TWO_PI), (0, 2.1), iron(pommel_tint, ("dagger:pommel",), rough=0.24), size=2.4, grid=70,
            along="u", tol=0.02, broken=0.02)
        add("button", forms.sphere(pom_base + 2.18 * a, 0.2), (0, TWO_PI), (-np.pi / 2, np.pi / 2),
            iron(lambda P, N, u, v: (IRON, np.full(len(P), 0.4), N), ("dagger:pommel",)), size=1.2, grid=14, along="u", dead=False)

    def groups(self):
        return [(self.sc.cam.depth(self.e), self.forms)]

    def register(self, studio):
        studio.softness_of["dagger"] = self.softness
        studio.hook("glaze", self.shadow)
        studio.hook("finish", self.finish)

    def shadow(self, studio):
        """The blade's own shadow: a thin dark wedge that starts exactly where the steel enters the
        wood and runs away from the window, widening and paling as it goes."""
        from atelier.brushwork import fill_field
        from atelier.stencil import Stencil
        sc = self.sc
        cv, cam, rng, px = sc.cv, sc.cam, sc.rng, sc.px
        lt = self.lt or sc.lt
        Lw = lt.light.L
        table = studio.regions["table"]
        ss = np.linspace(BURIED, BLADE, 40)
        down = lambda P: P - Lw * (P[:, 1] / Lw[1])[:, None]
        left = down(self.tip + ss[:, None] * self.a - half_width(ss)[:, None] * self.bw)
        right = down(self.tip + ss[:, None] * self.a + half_width(ss)[:, None] * self.bw)
        poly = cam.project(np.concatenate([left, right[::-1]]))
        wedge = Stencil.from_polys([poly], sc.W, sc.H)
        base = cam.project(self.e[None])[0]
        run = cam.project(down((self.tip + (BURIED + 6.0) * self.a)[None]))[0] - base
        ang = float(np.arctan2(run[1], run[0]))
        reach = np.linalg.norm(cam.project(down((self.tip + BLADE * self.a)[None]))[0] - base)

        def inside(xy):
            return wedge.inside(xy) & table.inside(xy)
        for r, dens, far in ((3.4, 0.8, 1.0), (2.4, 0.8, 0.5), (1.6, 0.9, 0.2)):
            fill_field(cv, wedge.bbox(), None, lambda xy: np.full(len(xy), ang),
                       Brush(radius=px(r), hardness=0.0, thick=0.4, pickup=0.0), rng, length=px(r * 7), spacing=px(r * 0.8),
                       inside=lambda xy, far=far: inside(xy) & (np.linalg.norm(xy - base, axis=1) < far * reach),
                       paint=Paint.glaze("#2a190c", dens), coverage=1.0)
        cv.dry_now()

    # ---- last touches
    def finish(self, studio):
        sc = self.sc
        cv, cam, rng, px = sc.cv, sc.cam, sc.rng, sc.px
        lt = self.lt or sc.lt
        e, a, bw, bn, tip = self.e, self.a, self.bw, self.bn, self.tip
        region = studio.regions.get("dagger")
        near = studio.form("dagger_bevel_l").region
        far = studio.form("dagger_bevel_r").region

        def visible(xy, reg):
            return reg.inside(xy, px(0.6))
        # old polishing scratches: many faint lines along the blade, light on the lit bevel, dark on the other
        fine = Batch(cv, Brush(radius=px(0.45), hardness=0.5, thick=0.3, pickup=0.1, press_radius=0.5))
        for _ in range(60):
            sgn = rng.choice([-1, 1])
            s0 = rng.uniform(BURIED + 0.2, BLADE - RICASSO - 1.0)
            ln = rng.uniform(0.6, 3.6)
            u = rng.uniform(0.12, 0.9)
            ss = np.linspace(s0, min(s0 + ln, BLADE - RICASSO), 8)
            hw = half_width(ss)
            P = tip + ss[:, None] * a + (sgn * hw * (1 - u))[:, None] * bw + 0.2 * u * bn
            xy = cam.project(P)
            reg = near if sgn < 0 else far
            if not visible(xy, reg).all():
                continue
            here = studio.seen(xy[4][None])[0]
            tone = np.clip(here * rng.choice([1.18, 0.82]), 0, 1)
            fine.add(xy, tone, np.sin(np.linspace(0.3, np.pi - 0.3, 8)), tscale=rng.uniform(0.2, 0.7))
        fine.flush()
        # the ridge: a soft continuous light; the lit edge: one thin bright line, lost toward the point, broken by nicks
        ss = np.linspace(BURIED + 0.05, BLADE - RICASSO, 110)
        mid = 0.05 * np.sin(ss * 0.55 + 0.6) + 0.025 * np.sin(ss * 1.7)
        ridge = cam.project(tip + ss[:, None] * a + mid[:, None] * bw + 0.2 * bn)
        ok = visible(ridge, region)
        for run_ in np.split(np.nonzero(ok)[0], np.nonzero(np.diff(np.nonzero(ok)[0]) > 1)[0] + 1):
            if len(run_) > 6:
                cv.stroke(ridge[run_], Paint.opaque("#a9aba6"), Brush(radius=px(0.6), hardness=0.3, thick=0.35, pickup=0.3),
                          (0.25 + 0.5 * np.sin(np.linspace(0, 9, len(ss))) ** 2)[run_])
        Lw = lt.light.L
        lit_sign = -1 if (Lw @ (-bw)) > (Lw @ bw) else 1
        edge = cam.project(tip + ss[:, None] * a + (lit_sign * half_width(ss))[:, None] * bw)
        nick = np.ones(len(ss))
        for c in rng.uniform(0.25, 0.9, 4):
            nick *= 1 - np.exp(-((np.linspace(0, 1, len(ss)) - c) / 0.012) ** 2)
        cv.dry_now()

        # ---- where it went into the oak
        def at_table(sign):
            vs_ = np.linspace(0.0, BURIED + 3.0, 500)
            pts = tip + vs_[:, None] * a + (sign * half_width(vs_))[:, None] * bw
            return pts[np.argmin(np.abs(pts[:, 1]))] * [1, 0, 1]
        p_l, p_r = at_table(-1), at_table(+1)
        along_grain = np.array([1.0, 0.0, 0.0])
        mid_p = 0.5 * (p_l + p_r)
        # the slot opens along the grain, a little longer than the blade is wide, with crushed fibres at its lips
        half = 0.5 * np.linalg.norm(p_r - p_l) + 0.12
        slot = np.array([mid_p - along_grain * half, mid_p + [0, 0, 0.04], mid_p + along_grain * half])
        cv.stroke(cam.project(slot) + [0, px(1.0)], Paint.opaque("#120b07"), Brush(radius=px(1.8), hardness=0.3, thick=1.2, press_radius=0.75),
                  [0.15, 1.0, 0.15])
        lift = Batch(cv, Brush(radius=px(0.7), hardness=0.5, thick=1.3, pickup=0.1, press_radius=0.6))
        for _ in range(5):
            b0 = mid_p + along_grain * rng.uniform(-0.7, 0.7) + np.array([0, 0, rng.choice([-1, 1]) * rng.uniform(0.05, 0.10)])
            b1 = b0 + np.array([rng.uniform(-0.14, 0.14), rng.uniform(0.03, 0.10), rng.uniform(-0.04, 0.04)])
            lift.add(cam.project(np.array([b0, b1])), np.array([0.62, 0.50, 0.34]) * rng.uniform(0.7, 1.1), [1.0, 0.3])
        lift.flush()
        cv.dry_now()


STUDIO_READY = True
