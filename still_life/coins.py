"""Scattered hammered coins: gold, silver and copper, struck by hand around 1600.

Nothing like modern money. Each is a thin flan (well under a millimetre for gold and silver),
cut out by eye so the outline wanders and has flats, struck off-centre so the design runs off
one side, and worn smooth on the high points. They lie where they slid: mostly flat, lapping
over one another like scales. Nine of them are gold.
"""
import numpy as np

from atelier import forms
from atelier.brushwork import Batch
from atelier.canvas import Brush, Paint
from atelier.forms import fbm3, noise3
from atelier.geom import Disc, Glow, frame, normalize, smoothstep
from atelier.shading import lin
from atelier.studio import Form
from . import scene as L

TWO_PI = 2 * np.pi
METALS = {
    #          tint (linear)                tarnish / grime        radius  thickness
    "gold":   (np.array([1.00, 0.70, 0.26]), lin("#6a3f1c"), 1.20, 0.10),
    "angel":  (np.array([1.00, 0.72, 0.28]), lin("#6a3f1c"), 1.45, 0.10),
    "shill":  (np.array([0.80, 0.80, 0.77]), lin("#2b2c33"), 1.55, 0.11),
    "six":    (np.array([0.80, 0.80, 0.77]), lin("#2b2c33"), 1.25, 0.10),
    "penny":  (np.array([0.78, 0.78, 0.75]), lin("#2b2c33"), 0.78, 0.08),
    "copper": (np.array([0.72, 0.36, 0.22]), lin("#3e5a48"), 1.05, 0.17),
}


def scatter():
    """Where each coin lies, relative to the middle of the spill: (x, z, lift, tilt-x, tilt-z, kind)."""
    lay = [
        # the heap: coins lying flat underneath, others slid on top of them and tipped at all angles
        (-1.9, 0.3, 0.00, 0.02, 0.02, "shill"), (0.6, 1.4, 0.00, -0.02, 0.03, "angel"), (0.2, -1.3, 0.00, 0.03, -0.02, "copper"),
        (2.3, 0.0, 0.00, -0.02, 0.01, "six"), (-0.9, 2.4, 0.00, 0.02, 0.04, "gold"),
        (-0.8, 0.1, 0.20, 0.16, 0.08, "gold"), (1.3, 0.6, 0.22, -0.20, 0.12, "shill"), (0.3, -0.5, 0.36, 0.08, -0.22, "copper"),
        (-0.3, 1.3, 0.34, 0.18, 0.24, "six"), (1.6, -0.9, 0.24, -0.26, -0.10, "gold"),
        (0.5, 0.3, 0.62, -0.34, 0.36, "gold"), (-1.0, 0.9, 0.60, -0.42, 0.46, "angel"), (0.9, 1.4, 0.44, 0.30, 0.34, "penny"),
        (-0.1, -0.2, 0.84, -0.10, 0.20, "shill"),
        # the trail toward the hourglass they were knocked from beside
        (-4.6, -1.6, 0.00, 0.02, 0.00, "gold"), (-3.4, -3.4, 0.00, 0.00, 0.03, "shill"), (-5.6, -4.6, 0.00, 0.01, 0.02, "gold"),
        (-2.2, -5.0, 0.00, 0.03, 0.01, "copper"),
        # strays toward the front, and the ones the skull's jaw came down beside
        (-4.2, 3.6, 0.00, 0.02, 0.02, "gold"), (-0.6, 5.2, 0.00, 0.01, 0.00, "angel"), (3.4, 4.3, 0.00, 0.00, 0.02, "penny"),
        (-6.2, 6.25, 0.00, 0.01, 0.01, "six"), (4.3, 5.7, 0.00, 0.03, 0.00, "gold"), (6.4, 5.2, 0.00, 0.00, 0.02, "copper"),
        (1.9, 3.6, 0.00, 0.02, 0.01, "shill"),
    ]
    return lay


def relief(a, b, kind, seed):
    """Height of what was struck into a face at (a, b) in units of the radius: a raised border, a
    ring of worn lettering and a device, all soft with wear and pushed off-centre, so part of the
    border has run off the edge altogether."""
    off = 0.10 * np.array([np.cos(seed * 2.4), np.sin(seed * 2.4)])
    a, b = a - off[0], b - off[1]
    r = np.hypot(a, b)
    ang = np.arctan2(b, a)
    border = smoothstep(0.74, 0.83, r) * (1 - smoothstep(0.90, 0.99, r))
    legend = smoothstep(0.46, 0.55, r) * (1 - smoothstep(0.66, 0.74, r)) * (0.5 + 0.5 * np.sin(ang * 9 + 2.2 * np.sin(ang * 2 + seed)))
    if kind in ("gold", "angel"):          # a shield quartered by a cross
        shield = 1 - smoothstep(0.30, 0.46, np.maximum(np.abs(a), np.abs(b + 0.04) * 0.85 + 0.3 * a * a))
        cross = np.minimum(1.0, (1 - smoothstep(0.03, 0.12, np.abs(a))) + (1 - smoothstep(0.03, 0.12, np.abs(b))))
        device = shield * (0.4 + 0.4 * cross)
    elif kind == "copper":
        device = 0.5 * (1 - smoothstep(0.10, 0.45, r * (1.0 + 0.4 * np.cos(ang * 4))))
    else:                                  # a crowned head in profile, as soft lumps
        head = 1 - smoothstep(0.5, 1.0, np.hypot((a - 0.02) / 0.30, (b - 0.06) / 0.36))
        crown = 1 - smoothstep(0.5, 1.0, np.hypot((a - 0.02) / 0.26, (b - 0.40) / 0.12))
        neck = 1 - smoothstep(0.5, 1.0, np.hypot((a - 0.06) / 0.20, (b + 0.38) / 0.20))
        device = 0.6 * np.maximum(head, np.maximum(crown, neck))
    return 0.6 * border + 0.4 * legend + device


class Coins:
    softness = (0.05, 0.0)
    reach = 12.0

    def __init__(self, sc):
        self.sc = sc
        self.lt = None
        self.coins = []
        for i, (x, z, lift, tx, tz, kind) in enumerate(scatter()):
            tint, grime, R, t = METALS[kind]
            c = L.COINS + [x, lift + t / 2, z]
            n = normalize([tx, 1.0, tz])
            e1, e2 = frame(n)
            rng = np.random.default_rng(100 + i)
            # the outline: cut by eye, with a flat or two and a wandering edge
            flats = [(rng.uniform(0, TWO_PI), rng.uniform(0.03, 0.09)) for _ in range(rng.integers(1, 4))]
            lump = rng.uniform(0.0, TWO_PI, 3)
            self.coins.append(dict(i=i, c=c, n=n, e1=e1, e2=e2, kind=kind, tint=tint, grime=grime, R=R, t=t,
                                   flats=flats, lump=lump, spin=rng.uniform(0, TWO_PI), bend=rng.normal(0, 0.05, 2)))
        self.forms = []
        self._build()

    @property
    def centre(self):
        return L.COINS + [0.0, 0.2, 1.0]

    def outline(self, k, ang):
        """Radius of coin k in direction ang (in units of its nominal radius)."""
        r = 1.0 + 0.025 * np.sin(2 * ang + k["lump"][0]) + 0.02 * np.sin(3 * ang + k["lump"][1]) + 0.012 * np.sin(5 * ang + k["lump"][2])
        for a0, depth in k["flats"]:
            d = np.abs((ang - a0 + np.pi) % TWO_PI - np.pi)
            r = r - depth * np.exp(-(d / 0.35) ** 2)
        return r

    def occluders(self):
        return [Disc("coin%d" % k["i"], k["c"] + k["n"] * k["t"] / 2, k["n"], k["R"] * 0.97) for k in self.coins]

    def glows(self):
        return [Glow("coinheap", L.COINS + [0.4, 0.3, 0.4], 2.2, np.array([0.55, 0.42, 0.22]))]

    def _build(self):
        sc = self.sc
        for k in self.coins:
            c, n, e1, e2, R = k["c"], k["n"], k["e1"], k["e2"], k["R"]
            top = c + n * k["t"] / 2

            def face(u, v, k=k, top=top, n=n, e1=e1, e2=e2, R=R):
                # laid out as a flat patch (u, v across the coin), so the brush goes straight across it
                u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
                x, y = R * u, R * v
                # a thin flan is never flat: it is slightly buckled
                dish = k["bend"][0] * u ** 2 * R * 0.5 + k["bend"][1] * u * v * R * 0.5
                P = top + x[..., None] * e1 + y[..., None] * e2 + dish[..., None] * n
                gx = k["bend"][0] * u + k["bend"][1] * v * 0.5
                gy = k["bend"][1] * u * 0.5
                N = normalize(n - gx[..., None] * e1 - gy[..., None] * e2)
                return P, N

            def within(P, N, u, v, k=k):
                u, v = np.asarray(u), np.asarray(v)
                return np.hypot(u, v) <= self.outline(k, np.arctan2(v, u))

            def col(P, N, u, v, k=k, top=top, e1=e1, e2=e2, R=R):
                lt = self.lt or sc.lt
                d = P - top
                x, y = (d @ e1) / R, (d @ e2) / R
                cs, sn = np.cos(k["spin"]), np.sin(k["spin"])
                a, b = cs * x + sn * y, -sn * x + cs * y
                eps = 0.03
                h0 = relief(a, b, k["kind"], k["i"])
                ha = (relief(a + eps, b, k["kind"], k["i"]) - h0) / eps
                hb = (relief(a, b + eps, k["kind"], k["i"]) - h0) / eps
                gx, gy = cs * ha - sn * hb, sn * ha + cs * hb
                Nb = normalize(N - 0.055 * (gx[:, None] * e1 + gy[:, None] * e2))
                low = 1 - np.clip(h0, 0, 1)
                mottle = smoothstep(0.35, 0.75, fbm3(P * 4.0 + k["i"], 1.0, 3, 101))
                if k["kind"] in ("gold", "angel"):       # gold does not tarnish: only reddish dirt in the low places
                    dirt = np.clip(0.20 + 0.30 * low * mottle, 0, 0.7)
                    matte = k["tint"] * 0.55 * (1 - 0.6 * (low * mottle)[:, None]) + k["grime"] * 0.6 * (low * mottle)[:, None]
                elif k["kind"] == "copper":              # copper goes brown and matte, with green only in the hollows
                    dirt = np.clip(0.58 + 0.27 * low, 0, 0.95)
                    green = (low * smoothstep(0.45, 0.8, mottle))[:, None]
                    matte = lin("#6b3f28") * (1 - 0.7 * green) + k["grime"] * 0.7 * green
                else:                                    # silver is grey, rubbed bright on the high points, toned dark in the low
                    dirt = np.clip(0.32 + 0.33 * low, 0, 0.9)
                    matte = k["tint"] * 0.50 * (1 - 0.7 * (low * (0.4 + 0.6 * mottle))[:, None]) + k["grime"] * 0.7 * (low * (0.4 + 0.6 * mottle))[:, None]
                return lt.mirror(k["tint"], P, Nb, skip=("coin%d" % k["i"],), rough=0.20, taps=5, dirt=(matte, dirt),
                                 rng=np.random.default_rng(len(P) + k["i"]))
            f = Form("coin%d" % k["i"], face, (-1.08, 1.08), (-1.08, 1.08), col, owner="coins", size=2.8, grid=30, along="u",
                     tol=0.025, broken=0.02, outline_grid=80, vis=within)
            f.coin = k
            k["form"] = f
            self.forms.append(f)

    def groups(self):
        cam = self.sc.cam
        # lower coins before the ones lying on them; otherwise far before near
        return [(cam.depth(k["c"]) - 40.0 * k["c"][1], [k["form"]]) for k in self.coins]

    def register(self, studio):
        studio.softness_of["coins"] = self.softness
        studio.hook("finish", self.finish)

    def finish(self, studio):
        """What makes each a slip of metal and not a patch of colour: the thickness of its edge where
        the edge faces us, the hairline of dark where it meets what it lies on, a broken line of light
        along the rim that faces the window, and one fat touch where the metal flashes."""
        sc = self.sc
        cv, cam, rng, px = sc.cv, sc.cam, sc.rng, sc.px
        lt = self.lt or sc.lt
        Lw = lt.light.L
        dark = Batch(cv, Brush(radius=px(1.0), hardness=0.35, thick=0.8, pickup=0.2, press_radius=0.5))
        light = Batch(cv, Brush(radius=px(0.9), hardness=0.5, thick=1.8, pickup=0.1, press_radius=0.6))
        fleck = Batch(cv, Brush(radius=px(1.6), hardness=0.5, thick=2.8, pickup=0.1, press_radius=0.6))
        walls = []
        ang = np.linspace(0, TWO_PI, 180, endpoint=False)
        # nothing of a coin's edge may be drawn over the strand lying across it, or over a coin lying on it
        over = np.zeros((sc.H, sc.W), bool)
        strand = sc.objs.get("strand")
        if strand is not None:
            from PIL import Image, ImageDraw
            im = Image.new("L", (sc.W, sc.H), 0)
            dr = ImageDraw.Draw(im)
            for pts, hidden in strand.pieces():
                if not hidden:
                    q = cam.project(pts)
                    w = int(max(2, round(2.6 * 0.115 * float(np.median(cam.px_per_cm(pts))))))
                    dr.line([tuple(p) for p in q], fill=255, width=w)
            over |= np.asarray(im) > 0
        order = {f.name: i for i, f in enumerate(studio.forms)}

        def clear(q, k):
            x = np.clip(np.round(q[:, 0] - 0.5).astype(int), 0, sc.W - 1)
            y = np.clip(np.round(q[:, 1] - 0.5).astype(int), 0, sc.H - 1)
            ok = ~over[y, x]
            mine = order[k["form"].name]
            for o in self.coins:
                if order[o["form"].name] > mine:
                    ok &= ~o["form"].region.inside(q, 0)
            return ok
        for k in self.coins:
            region = k["form"].region
            if region.area() < 60:
                continue
            top = k["c"] + k["n"] * k["t"] / 2
            radial = np.cos(ang)[:, None] * k["e1"] + np.sin(ang)[:, None] * k["e2"]
            rim = top + (k["R"] * self.outline(k, ang))[:, None] * radial
            low = rim - k["n"] * k["t"]
            xy, xy_low = cam.project(rim), cam.project(low)
            ctr = cam.project(top[None])[0]
            # this bit of rim is seen only if the face just inside it is this coin's and not one lying over it
            seen = region.inside(xy + (ctr - xy) * 0.06, 0) & clear(xy, k) & clear(xy_low, k)
            faces_us = np.sum(radial * cam.to_eye(rim), axis=1) > 0.05
            skip = ("coin%d" % k["i"],)
            wall_col = lt.mirror(k["tint"], 0.5 * (rim + low), radial, skip=skip, rough=0.25, taps=5,
                                 dirt=(k["tint"] * 0.45, np.full(len(ang), 0.45)))
            lit = lt.light.lit(rim + k["n"] * 0.02, skip)
            wall_col = wall_col * (0.30 + 0.70 * lit)[:, None]
            h = np.linalg.norm(xy_low - xy, axis=1)
            idx = np.nonzero(seen & faces_us)[0]
            for run in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
                if len(run) < 3:
                    continue
                mid = 0.5 * (xy[run] + xy_low[run])
                walls.append((mid, wall_col[run].mean(axis=0), max(px(0.6), 0.5 * float(h[run].mean()))))
                # the dark line under the edge, where it meets the table or the coin beneath
                dark.add(xy_low[run] + [0.0, px(0.5)], np.array([0.040, 0.027, 0.018]),
                         np.sin(np.linspace(0.25, np.pi - 0.25, len(run))) ** 0.5)
            # light along the rim toward the window: broken, never a drawn circle
            toward = (radial @ Lw > 0.25) & seen & (lit > 0.4)
            hi = np.clip(lt.tone(k["tint"] * 1.9 + 0.25), 0, 1)
            idx = np.nonzero(toward & (rng.random(len(ang)) > 0.22))[0]
            for run in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
                if len(run) > 2:
                    light.add(xy[run] + (ctr - xy[run]) * 0.03, hi, np.sin(np.linspace(0.3, np.pi - 0.3, len(run))),
                              tscale=float(lit[run].mean()))
            if toward.any() and k["kind"] != "copper" and rng.random() < 0.85:
                i0 = np.nonzero(toward)[0][np.argmax((radial @ Lw)[toward])]
                q = xy[i0] + (ctr - xy[i0]) * 0.05
                t2 = xy[(i0 + 3) % len(ang)] - xy[i0 - 3]
                t2 /= max(np.linalg.norm(t2), 1e-9)
                fleck.add(np.array([q - t2 * px(2.0), q + t2 * px(2.0)]), np.clip(hi * 1.08 + 0.06, 0, 1), [0.6, 1.0],
                          tscale=float(lit[i0]))
        for mid, colr, r in walls:
            cv.stroke(mid, Paint.opaque(np.clip(colr, 0, 1)), Brush(radius=r, hardness=0.6, thick=1.2, pickup=0.3), 1.0)
        cv.dry_now()
        dark.flush()
        cv.dry_now()
        light.flush()
        fleck.flush()
        cv.dry_now()


STUDIO_READY = True
