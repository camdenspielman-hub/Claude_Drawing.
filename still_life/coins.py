"""A small heap of mixed coins: gold, silver and copper, worn, lying as they fell."""
import numpy as np

from atelier import forms
from atelier.brushwork import paint_form
from atelier.forms import fbm3
from atelier.geom import Disc, frame, normalize, smoothstep
from atelier.shading import lin
from atelier.stencil import Stencil
from . import scene as L

TWO_PI = 2 * np.pi
METALS = {
    #          tint (linear)              tarnish colour       radius  thickness
    "gold":   (np.array([1.00, 0.72, 0.28]), lin("#6b4a1c"), 1.15, 0.14),
    "silver": (np.array([0.86, 0.85, 0.80]), lin("#4a4640"), 1.45, 0.20),
    "small":  (np.array([0.86, 0.85, 0.80]), lin("#4a4640"), 1.00, 0.13),
    "copper": (np.array([0.90, 0.48, 0.32]), lin("#3d4f3c"), 1.30, 0.17),
}


def heap():
    """Where each coin lies: (centre offset from the heap's middle, facing, metal).
    Nine of them are gold."""
    rng = np.random.default_rng(23)
    out = []
    # the bed of the heap, lying nearly flat and overlapping
    ring = [(-3.4, 0.6, "copper"), (-1.2, 1.5, "gold"), (1.3, 1.1, "silver"), (3.3, -0.2, "gold"),
            (2.2, -1.9, "copper"), (-0.2, -1.6, "silver"), (-2.6, -1.5, "gold"), (0.3, 0.0, "copper")]
    for i, (x, z, m) in enumerate(ring):
        tilt = normalize([rng.normal(0, 0.10), 1.0, rng.normal(0, 0.10)])
        out.append((np.array([x, 0.10 + 0.05 * (i % 3), z]), tilt, m))
    # coins that slid onto them, tipped where they rest on an edge
    for x, z, y, tx, tz, m in [(-2.2, 0.2, 0.42, 0.22, 0.10, "gold"), (0.9, -0.9, 0.46, -0.20, 0.16, "gold"),
                               (2.4, 0.5, 0.44, 0.18, -0.20, "small"), (-0.6, 1.0, 0.50, -0.12, -0.24, "silver"),
                               (-1.1, -0.6, 0.78, 0.26, 0.20, "copper"), (1.0, 0.3, 0.82, -0.28, 0.12, "gold"),
                               (0.0, -0.2, 1.12, 0.10, 0.30, "small")]:
        out.append((np.array([x, y, z]), normalize([tx, 1.0, tz]), m))
    # one propped steeply against the heap, and strays
    out.append((np.array([-4.3, 0.78, -0.6]), normalize([-0.80, 0.55, 0.22]), "gold"))
    out.append((np.array([4.9, 0.09, 1.9]), normalize([0.02, 1.0, 0.01]), "copper"))
    out.append((np.array([-5.6, 0.07, 2.6]), normalize([0.0, 1.0, 0.02]), "small"))
    out.append((np.array([5.8, 0.08, -1.4]), normalize([0.01, 1.0, 0.0]), "gold"))
    out.append((np.array([1.6, 0.09, 3.4]), normalize([0.0, 1.0, 0.0]), "silver"))
    out.append((np.array([-2.6, 0.08, 4.55]), normalize([0.02, 1.0, 0.0]), "gold"))      # just over the table's edge
    return out


def relief(a, b, kind):
    """Height of the struck design at face coordinates (a, b) in units of the radius: a raised rim,
    a ring of lettering, and a device in the middle. Worn, so everything is low and soft."""
    r = np.hypot(a, b)
    ang = np.arctan2(b, a)
    h = 0.9 * smoothstep(0.86, 0.93, r)                                     # the rim
    letters = smoothstep(0.66, 0.70, r) * (1 - smoothstep(0.79, 0.83, r))
    h = h + 0.45 * letters * (0.5 + 0.5 * np.sign(np.sin(ang * (19 if kind != "small" else 13) + 3 * np.sin(ang * 3))))
    if kind == "gold":         # a shield with a cross
        shield = (1 - smoothstep(0.42, 0.47, np.maximum(np.abs(a), np.abs(b + 0.05) * 0.85 + 0.25 * a * a)))
        cross = np.minimum(1.0, (1 - smoothstep(0.05, 0.08, np.abs(a))) + (1 - smoothstep(0.05, 0.08, np.abs(b + 0.02))))
        h = h + shield * (0.35 + 0.35 * cross)
    elif kind == "copper":     # a rough star
        h = h + 0.55 * (1 - smoothstep(0.18, 0.5, r * (1.0 + 0.45 * np.cos(ang * 5))))
    else:                      # a head in profile: skull, face and neck as soft lumps
        head = 1 - smoothstep(0.75, 1.0, np.hypot((a - 0.02) / 0.30, (b - 0.12) / 0.36))
        face = 1 - smoothstep(0.7, 1.0, np.hypot((a + 0.20) / 0.16, (b - 0.02) / 0.22))
        neck = 1 - smoothstep(0.7, 1.0, np.hypot((a - 0.06) / 0.20, (b + 0.36) / 0.20))
        h = h + 0.6 * np.maximum(head, np.maximum(face, neck))
    return h


class Coins:
    def __init__(self, sc):
        self.sc = sc
        self.coins = []
        for off, n, m in heap():
            tint, tarnish, radius, thick = METALS[m]
            self.coins.append(dict(c=L.COINS + off, n=n, m=m, tint=tint, tarnish=tarnish, R=radius, t=thick))
        # lower coins first, then nearer ones over farther
        cam = sc.cam
        self.coins.sort(key=lambda k: (round(k["c"][1] / 0.3), -cam.depth(k["c"])))
        for i, k in enumerate(self.coins):
            k["i"] = i
            k["face"] = forms.disc(k["c"] + k["n"] * k["t"] / 2, k["n"], k["R"])
            k["edge"] = forms.revolution(k["c"] - k["n"] * k["t"] / 2, k["n"], lambda v, R=k["R"]: np.full_like(v, R))
            k["e1"], k["e2"] = frame(k["n"])
            k["spin"] = (i * 2.399) % TWO_PI

    def occluders(self):
        return [Disc("coin%d" % k["i"], k["c"] + k["n"] * k["t"] / 2, k["n"], k["R"]) for k in self.coins]

    def ground_ao(self, P):
        ao = np.ones(len(P))
        for k in self.coins:
            d = np.hypot(P[:, 0] - k["c"][0], P[:, 2] - k["c"][2])
            ao = ao * (1 - 0.55 * (1 - smoothstep(k["R"] * 0.6, k["R"] * 1.25, d)))
        return ao

    def prepare(self):
        sc, cam = self.sc, self.sc.cam
        self.stencil = Stencil.empty(sc.W, sc.H)
        for k in self.coins:
            k["st_face"] = Stencil.from_surface(cam, k["face"], (0, TWO_PI), (0, 1), sc.W, sc.H, grid=72)
            k["st_edge"] = Stencil.from_surface(cam, k["edge"], (0, TWO_PI), (0, k["t"]), sc.W, sc.H, grid=96)
            self.stencil = self.stencil | k["st_face"] | k["st_edge"]
        return self.stencil

    def paint(self):
        sc = self.sc
        cv, lt = sc.cv, sc.lt
        if "st_face" not in self.coins[0]:
            self.prepare()
        for k in self.coins:
            skip = ("coin%d" % k["i"],)
            c, n, R = k["c"], k["n"], k["R"]
            e1, e2 = k["e1"], k["e2"]
            cs, sn = np.cos(k["spin"]), np.sin(k["spin"])

            def grime(P, k=k):
                g = fbm3(P * 5.0 + k["i"], 1.0, 3, 101)
                return smoothstep(0.38, 0.72, g)

            def face_col(P, N, u, v, k=k, e1=e1, e2=e2, cs=cs, sn=sn):
                d = P - (k["c"] + k["n"] * k["t"] / 2)
                x, y = (d @ e1) / k["R"], (d @ e2) / k["R"]
                a, b = cs * x + sn * y, -sn * x + cs * y          # each coin is turned its own way
                eps = 0.012
                h0 = relief(a, b, k["m"])
                ha = (relief(a + eps, b, k["m"]) - h0) / eps
                hb = (relief(a, b + eps, k["m"]) - h0) / eps
                gx, gy = cs * ha - sn * hb, sn * ha + cs * hb
                Nb = normalize(N - 0.035 * (gx[:, None] * e1 + gy[:, None] * e2))
                heavy = 0.55 if k["m"] == "copper" else 0.35
                dirt = np.clip(0.08 + heavy * grime(P) + 0.30 * (1 - np.clip(h0, 0, 1)), 0, 0.85)   # grime sits in the low parts
                return lt.metal(k["tint"], P, Nb, skip=skip, gloss=5.0, shine=1.9, env=1.25, dirt=(k["tarnish"], dirt))

            def edge_col(P, N, u, v, k=k):
                reed = 0.5 + 0.5 * np.sin(u * 46)
                dirt = np.clip(0.15 + 0.35 * grime(P) + 0.30 * reed, 0, 0.85)
                return lt.metal(k["tint"], P, N, skip=skip, gloss=5.0, shine=1.6, env=1.2, dirt=(k["tarnish"], dirt))
            paint_form(sc, k["edge"], (0, TWO_PI), (0, k["t"]), edge_col, k["st_edge"], r1=1.8, r2=0, along="u",
                       grid=48, blend=0, r_detail=0.9, coverage=1.6)
            paint_form(sc, k["face"], (0, TWO_PI), (0, 1), face_col, k["st_face"], r1=3.2, r2=1.5, along="u",
                       grid=48, blend=0, r_detail=1.0, tol=0.02, coverage=1.3)
            cv.dry_now()
            # the raised rim turns to the window along part of its round: a thin, thick-laid light
            top = k["c"] + k["n"] * k["t"] / 2
            Lp = normalize(lt.light.L - (lt.light.L @ k["n"]) * k["n"])
            a0 = np.arctan2(Lp @ e2, Lp @ e1)
            if lt.light.lit(top[None], skip)[0] > 0.5:
                ang = a0 + np.pi + np.linspace(-0.75, 0.75, 22)        # the far side of the rim faces the light
                arc = top + 0.905 * R * (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2)
                hi = np.clip(lt.tone(k["tint"] * 1.5 + 0.25), 0, 1)
                from atelier.canvas import Brush, Paint
                cv.stroke(sc.cam.project(arc), Paint.opaque(hi), Brush(radius=sc.px(0.85), hardness=0.5, thick=1.8,
                          pickup=0.1, press_radius=0.6), np.sin(np.linspace(0, np.pi, 22)) ** 0.7)
            cv.dry_now()
