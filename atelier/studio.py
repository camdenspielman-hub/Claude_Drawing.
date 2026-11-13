"""The studio: how a whole picture is brought along together, in the old order of work.

    draw     thin umber lines: where everything goes
    dead     the 'dead colouring': a monochrome underpainting of the big lights and darks
    colour   every form laid in with its colour, thin in the darks and fat in the lights
    model    smaller brushes: the turning of each form, the edges of shadows
    detail   the point of the brush where it is needed; hard edges restated, soft ones left
    glaze    transparent darks drawn over the shadows to deepen and join them
    finish   highlights, accents, and whatever is particular to each object

Every stage covers the whole canvas while its paint is wet, so neighbouring things can be
worked into one another; only then does it dry and the next stage begin.
"""
import numpy as np

from .brushwork import Batch, paint_surface, paint_solid, fill_field
from .canvas import Brush, Paint
from .geom import smoothstep
from .shading import luma
from .stencil import Stencil

STAGES = ("draw", "dead", "colour", "model", "detail", "glaze", "finish")
BROAD = 1.5          # how much broader the first lay-in of colour is than each form's nominal brush
UMBER_DARK = np.array([0.105, 0.070, 0.045])
UMBER_LIGHT = np.array([0.80, 0.73, 0.60])


def monochrome(rgb):
    """A colour's place on the scale from raw umber to warm white."""
    y = luma(rgb)[:, None]
    return UMBER_DARK + (UMBER_LIGHT - UMBER_DARK) * y ** 0.9


class Form:
    """One paintable form: a surface, its colour as a function of place, and how it is to be handled."""

    def __init__(self, name, surf, ur, vr, colour, *, owner="", along="u", grid=40, size=5.0, vis=None,
                 min_facing=0.0, finish=1.0, tol=0.03, coverage=1.2, opaque=True, broken=0.03,
                 outline_grid=None, dead=True, body=5.0, thick=(0.40, 1.25), flat=None, stencil=None, flow=None):
        self.name, self.surf, self.ur, self.vr, self.colour = name, surf, ur, vr, colour
        self.owner = owner or name
        self.along = along
        self.grid = grid
        self.size = size                # radius of the lay-in brush, in full-canvas px
        self.vis = vis
        self.min_facing = min_facing
        self.finish = finish            # 0..1: how far this form is carried (less for what should stay quiet)
        self.tol = tol
        self.coverage = coverage
        self.opaque = opaque            # False: seen through (glass); things behind it are still painted
        self.broken = broken
        self.outline_grid = outline_grid or max(48, grid * 2)
        self.dead = dead
        self.body = body
        self.thick = thick
        self.flat = flat                # for a flat form: (origin, eu, ev) of its plane, so any pixel can be traced back to it
        self.stencil = stencil          # the outline, if it is already known (a sculpted solid draws its own)
        self.flow = flow                # for a sculpted solid: flow(P, N, which) -> the way strokes run over it
        self.region = None


class Studio:
    def __init__(self, sc, focus=None):
        self.sc = sc
        self.forms = []
        self.hooks = {s: [] for s in STAGES}
        self.focus = focus              # canvas point the eye should go to; edges soften away from it
        self.softness_of = {}           # owner -> (base softness of its edges, extra for being far back)
        self.no_glaze = ("wall",)       # owners the shadow glaze leaves alone
        self._look = None

    # ---- setting up
    def add(self, *forms):
        self.forms.extend(forms)

    def hook(self, stage, fn):
        """Something particular to do at the end of a stage: fn(studio)."""
        self.hooks[stage].append(fn)

    def prepare(self):
        """Draw every outline, then work out what of each form is actually seen."""
        sc = self.sc
        for f in self.forms:
            if f.stencil is None:
                f.stencil = Stencil.from_surface(sc.cam, f.surf, f.ur, f.vr, sc.W, sc.H, grid=f.outline_grid,
                                                 visible=f.vis, min_facing=f.min_facing)
        covered = np.zeros((sc.H, sc.W), bool)
        for f in reversed(self.forms):
            f.region = Stencil(f.stencil.mask & ~covered)
            if f.opaque:
                covered |= f.stencil.mask
        self.regions = {}
        for f in self.forms:
            m = self.regions.get(f.owner)
            self.regions[f.owner] = f.region.mask.copy() if m is None else (m | f.region.mask)
        self.regions = {k: Stencil(v) for k, v in self.regions.items()}

    def form(self, name):
        return next(f for f in self.forms if f.name == name)

    # ---- looking at the canvas, as a painter does before deciding an edge
    def look(self):
        if self._look is None:
            self._look = self.sc.cv.array()
        return self._look

    def seen(self, xy):
        a = self.look()
        x = np.clip(np.round(xy[..., 0] - 0.5).astype(np.int64), 0, a.shape[1] - 1)
        y = np.clip(np.round(xy[..., 1] - 0.5).astype(np.int64), 0, a.shape[0] - 1)
        return a[y, x]

    def dry(self):
        self.sc.cv.dry_now()
        self._look = None

    # ---- edges: lost and found
    def softness(self, region, xy, owner):
        """0 = keep this bit of outline crisp, 1 = let it go entirely. An edge is kept where light
        meets dark near the centre of interest, and lost where like meets like, in shadow, or far off."""
        px = self.sc.px
        base, far = self.softness_of.get(owner, (0.15, 0.0))
        n = region.inward(xy)
        d = region.depth(xy)[:, None]
        cin = self.seen(xy + n * px(4.0))
        cout = self.seen(xy - n * (d + px(4.0)))
        contrast = np.abs(luma(cin) - luma(cout)) + 0.35 * np.mean(np.abs(cin - cout), axis=1)
        s = base + far + 0.60 * (1 - smoothstep(0.035, 0.20, contrast))
        s = s + 0.20 * smoothstep(0.22, 0.08, np.maximum(luma(cin), luma(cout)))
        if self.focus is not None:
            away = np.linalg.norm(xy - np.asarray(self.focus), axis=1) / self.sc.W
            s = s + 0.22 * smoothstep(0.16, 0.55, away)
        return np.clip(s, 0, 1)

    def soften_edges(self, strength=1.0, owners=None):
        """Work a clean soft brush along the outlines that should be soft, while the paint is wet."""
        sc = self.sc
        cv, rng, px = sc.cv, sc.rng, sc.px
        self._look = None
        batch = Batch(cv, Brush(kind="blender", radius=px(1.0), hardness=0.15, pickup=0.38, load=0.5))
        for owner, region in self.regions.items():
            if owners is not None and owner not in owners:
                continue
            pts = region.edge_points(px(2.6), rng)
            if len(pts) == 0:
                continue
            s = self.softness(region, pts, owner)
            n = region.inward(pts)
            t = np.stack([-n[:, 1], n[:, 0]], axis=1)
            for i in np.nonzero(s > 0.30)[0]:
                ln = px(6.0 + 12.0 * s[i])
                c = pts[i] - n[i] * px(0.8)
                batch.add(np.array([c - t[i] * ln / 2, c, c + t[i] * ln / 2]), np.zeros(3), 1.0,
                          1.4 + 8.0 * s[i] * strength)
        batch.flush(rng=rng)
        self._look = None

    # ---- the stages
    def _lay(self, f, r, along, **kw):
        sc = self.sc
        px = sc.px
        brush = kw.pop("brush", None) or Brush(radius=px(r), hardness=kw.pop("hardness", 0.5), thick=kw.pop("thick", 1.3),
                                               pickup=kw.pop("pickup", 0.5))
        if f.flow is not None:
            return paint_solid(sc.cv, sc.cam, f.surf, f.colour, brush, sc.rng, region=f.region, flow=f.flow, which=along,
                               length=px(kw.pop("length", r * 4.5)), spacing=px(kw.pop("spacing", r * 0.7)),
                               visible=kw.pop("visible", f.vis), body=f.body, **kw)
        return paint_surface(sc.cv, sc.cam, f.surf, f.colour, brush, sc.rng, urange=f.ur, vrange=f.vr, along=along,
                             length=px(kw.pop("length", r * 4.5)), spacing=px(kw.pop("spacing", r * 0.7)),
                             stencil=f.region, visible=kw.pop("visible", f.vis), grid=f.grid, min_facing=f.min_facing,
                             body=f.body, **kw)

    def _fit(self, f, r):
        """A brush no bigger than the form can take."""
        a = f.region.area() * (1.0 / max(self.sc.s, 1e-6)) ** 2
        return float(np.clip(min(r, np.sqrt(max(a, 1.0)) / 3.2), 0.8, r))

    def stage_draw(self):
        sc = self.sc
        cv, rng, px = sc.cv, sc.rng, sc.px
        line = Batch(cv, Brush(radius=px(1.1), hardness=0.3, thick=0.22, pickup=0.2, press_radius=0.5))
        ink = Paint.opaque("#4a3725")           # thin umber, a little darker than the ground: a guide, to be painted out
        for f in self.forms:
            if f.region.area() < 30:
                continue
            pts = f.region.edge_points(px(5.0), rng)
            if len(pts) == 0:
                continue
            n = f.region.inward(pts)
            t = np.stack([-n[:, 1], n[:, 0]], axis=1)
            for i in range(len(pts)):
                ln = px(rng.uniform(9, 20))
                wob = n[i] * px(rng.normal(0, 0.6))
                line.add(np.array([pts[i] - t[i] * ln / 2, pts[i] + wob, pts[i] + t[i] * ln / 2]), None,
                         [0.3, 1.0, 0.3], paint=ink)
        line.flush(rng=rng)
        self.dry()

    def stage_dead(self):
        for f in self.forms:
            if not f.dead or f.finish < 0 or f.region.area() < 60:
                continue
            r = self._fit(f, f.size * 1.5)
            self._lay(f, r, f.along, thick=0.85, hardness=0.3, margin=self.sc.px(0.3 * r), coverage=f.coverage,
                      recolour=monochrome, broken=0.03)
        self.soften_edges(strength=1.3)
        self.dry()

    def stage_colour(self):
        for f in self.forms:
            if f.finish < 0:
                continue
            r = self._fit(f, f.size * BROAD)
            # each form keeps to its own shape, so wet neighbours are not dragged into one another by accident
            self._lay(f, r, f.along, margin=self.sc.px(0.9 * r), coverage=f.coverage * 1.1, thick_by_value=f.thick,
                      broken=f.broken)
            if f.flat is not None:
                self._to_border(f, r)
        self.soften_edges(strength=1.0)
        self.dry()

    def _to_border(self, f, r):
        """Strokes along the canvas's own edges, so a flat form that runs off the picture leaves no bare ground there."""
        sc = self.sc
        cv, cam, rng = sc.cv, sc.cam, sc.rng
        px = sc.px
        origin, eu, ev = f.flat
        nrm = np.cross(eu, ev)
        rr = px(min(r, 12.0))
        batch = Batch(cv, Brush(radius=rr, hardness=0.4, thick=1.2, pickup=0.5), f.body)
        W, H = sc.W, sc.H
        n = 0
        for a, b in (((0, 0), (W, 0)), ((W, 0), (W, H)), ((W, H), (0, H)), ((0, H), (0, 0))):
            a, b = np.array(a, float), np.array(b, float)
            d = (b - a) / np.linalg.norm(b - a)
            inward = np.array([-d[1], d[0]])
            m = int(np.linalg.norm(b - a) / (rr * 0.8)) + 1
            for off in (0.6, 1.8):
                pts = a + np.linspace(0, 1, m)[:, None] * (b - a) + inward * rr * off + rng.normal(0, rr * 0.15, (m, 2))
                pts = np.clip(pts, [0.5, 0.5], [W - 0.5, H - 0.5])
                keep = f.region.inside(pts, 0)
                if not keep.any():
                    continue
                pts = pts[keep]
                P = cam.onto_plane(pts, origin, nrm)
                dd = P - origin
                u, v = (dd @ eu) / (eu @ eu), (dd @ ev) / (ev @ ev)
                Ps, N = f.surf(u, v)
                cols = f.colour(Ps, N, u, v)
                for i in range(len(pts)):
                    batch.add(np.array([pts[i] - d * rr * 1.6, pts[i] + d * rr * 1.6]), cols[i], [0.7, 1.0])
                    n += 1
        if n:
            batch.flush(rng=rng)

    def stage_model(self):
        sc = self.sc
        px = sc.px
        for f in self.forms:
            if f.finish < 0.25:
                continue
            other = "u" if f.along == "v" else "v"
            r2 = self._fit(f, max(1.6, f.size * 0.62))
            self._lay(f, r2, other, margin=px(0.7 * r2), hardness=0.55, thick=1.2, pickup=0.55, length=r2 * 5,
                      spacing=r2 * 0.75, thick_by_value=f.thick, broken=f.broken * 0.7)
            # wherever light gives way to shadow within a brush-length, short touches draw the edge of it
            r3 = max(1.0, 0.75 * r2)
            self._lay(f, r3, other, margin=px(0.5 * r3), hardness=0.5, thick=1.2, pickup=0.6, length=r3 * 3.0,
                      spacing=r3 * 0.8, detail=(f.tol, px(11.0)))
            # melt the strokes together, but only in the quiet passages
            self._lay(f, 3.0, f.along, brush=Brush(kind="blender", radius=px(min(3.4, r2 * 1.2)), hardness=0.2,
                                                   pickup=0.20, load=0.5),
                      length=10.0, spacing=6.5, paint=Paint.opaque("#000000"), detail=(f.tol * 0.8, px(10.0), "calm"))
        self.soften_edges(strength=0.6)
        self.dry()

    def stage_detail(self):
        sc = self.sc
        px = sc.px
        for f in self.forms:
            if f.finish < 0.5:
                continue
            rd = 1.3 if f.size > 2.5 else 0.9
            self._lay(f, rd, f.along, margin=px(0.5 * rd), hardness=0.55, thick=1.2, pickup=0.6, length=rd * 3.2,
                      spacing=rd * 0.85, detail=(f.tol * 1.4, px(3.5)))
        for f in self.forms:
            if f.finish >= 0 and getattr(f, "cut_in", True):
                self._cut_in(f)
        self.dry()
        self.correct()
        self.dry()

    def correct(self, tol=0.075, rounds=2):
        """Stand back and compare: wherever what is on the canvas differs from what should be there,
        go back in with a small brush. (Earlier passes only work where the form itself changes; this
        one looks at the picture.)"""
        sc = self.sc
        px = sc.px
        for k in range(rounds):
            self._look = None
            look = self.look()

            def needed(xy, cols, t=tol * (1.0 if k == 0 else 0.8)):
                return np.abs(self.seen(xy) - cols).max(axis=1) > t
            for f in self.forms:
                if f.finish < 0.5 or f.region.area() < 40:
                    continue
                r = 1.6 if k == 0 else 1.1
                self._lay(f, r, f.along, margin=px(0.5 * r), hardness=0.5, thick=1.2, pickup=0.5, length=r * 3.2,
                          spacing=r * 0.9, needed=needed, coverage=1.1)
            self.dry()

    def _cut_in(self, f):
        """Carry a form's paint right up to its outline with a small brush, following the edge."""
        sc = self.sc
        cv, cam, rng, px = sc.cv, sc.cam, sc.rng, sc.px
        region = f.region
        if region.area() < 40:
            return
        band = px(max(3.5, min(0.75 * f.size, 9.0)))
        if f.flat is not None:
            # a flat form: take points along the outline and ask the surface what colour belongs there
            origin, eu, ev = f.flat
            nrm = np.cross(eu, ev)
            steps = [(2.2, band), (1.0, px(2.6))]
            if f.finish < 0.25:          # a broadly painted form was left well short of its edges: close the gap in stages
                steps = [(f.size * 0.33, px(f.size * 1.0)), (f.size * 0.14, px(f.size * 0.45))] + steps
            for r, depth in steps:
                pts = region.edge_points(px(r * 0.55), rng, band=depth)
                if len(pts) == 0:
                    continue
                pts = pts[region.depth(pts) > px(r * 0.5)]
                if len(pts) == 0:
                    continue
                P = cam.onto_plane(pts, origin, nrm)
                d = P - origin
                u, v = (d @ eu) / (eu @ eu), (d @ ev) / (ev @ ev)
                Ps, N = f.surf(u, v)
                cols = f.colour(Ps, N, u, v)
                n = region.inward(pts)
                t = np.stack([-n[:, 1], n[:, 0]], axis=1)
                batch = Batch(cv, Brush(radius=px(r), hardness=0.6, thick=1.2, pickup=0.55), f.body)
                ln = px(r * 3.0)
                for i in range(len(pts)):
                    batch.add(np.array([pts[i] - t[i] * ln / 2, pts[i] + t[i] * ln / 2]), cols[i], [0.6, 1.0])
                batch.flush(rng=rng)
        elif region.area() < px(1000.0) ** 2:
            def near_edge(P, N, u, v):
                ok = region.depth(cam.project(P)) < band
                return ok if f.vis is None else ok & f.vis(P, N, u, v)
            for r in (2.0, 1.0):
                if r > f.size * 0.6:
                    continue
                self._lay(f, r, f.along, margin=px(0.5 * r), hardness=0.65, thick=1.2, pickup=0.6, length=r * 5.0,
                          spacing=r * 0.9, visible=near_edge)

    def stage_glaze(self):
        """A transparent warm dark drawn over everything already in shadow: it deepens the darks
        and lets neighbouring shadows run together, whatever objects they belong to."""
        sc = self.sc
        cv, rng, px = sc.cv, sc.rng, sc.px
        look = self.look()

        keep_off = [self.regions[k] for k in self.no_glaze if k in self.regions]

        def in_shadow(xy):
            ok = luma(self.seen(xy)) < 0.17
            for r in keep_off:
                ok &= ~r.inside(xy)
            return ok
        glaze = Paint.glaze("#3b2413", 1.0)
        for r, th in ((22.0, 0.22), (11.0, 0.14)):
            fill_field(cv, (0, 0, sc.W, sc.H), None, lambda xy: 0.3 * np.sin(xy[:, 0] / px(170.0)),
                       Brush(radius=px(r), hardness=0.0, thick=th, pickup=0.0), rng, length=px(r * 5),
                       spacing=px(r * 0.28), inside=in_shadow, paint=glaze, coverage=1.6)
        self.dry()

    def run(self, stages=STAGES, report=None):
        for s in stages:
            getattr(self, "stage_" + s, lambda: None)()
            for fn in self.hooks[s]:
                fn(self)
            self.dry()
            if report:
                report(s)
