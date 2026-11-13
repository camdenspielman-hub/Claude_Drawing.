"""The setting: a dark wall, and an old oak table top with its front edge."""
import numpy as np

from atelier import forms
from atelier.brushwork import paint_surface, fill_field
from atelier.canvas import Brush, Paint
from atelier.forms import fbm3, noise3
from atelier.geom import smoothstep
from atelier.shading import lin
from . import scene as L

OAK = lin("#5c4630")
OAK_DARK = lin("#3d2c1c")


def window_pool(P):
    """Where the window's light falls: fullest front and centre-left, fading to the back and sides."""
    x, z = P[:, 0], P[:, 2]
    side = np.exp(-((x + 1.0) / 30.0) ** 2)
    depth = smoothstep(-26.0, 4.0, z)
    return 0.16 + 0.84 * side * (0.25 + 0.75 * depth)


def wall_colour(sc):
    W, H = sc.W, sc.H
    x0, y0 = sc.crop[0], sc.crop[1]

    def col(xy):
        # work in full-canvas coordinates so a cropped study matches the whole picture
        X = (xy[:, 0] / sc.s + x0) / L.W0
        Y = (xy[:, 1] / sc.s + y0) / L.H0
        glow = np.exp(-(((X - 0.70) / 0.38) ** 2 + ((Y - 0.34) / 0.42) ** 2))
        v = 0.030 + 0.105 * glow
        q = np.stack([X * 7.0, Y * 7.0, 0 * X], 1)
        v = v * (0.82 + 0.36 * fbm3(q, 1.0, 3, 41))
        warm = 0.5 + 0.5 * fbm3(q * 0.6 + 9.0, 1.0, 2, 43)
        return np.stack([v * (1.06 + 0.10 * warm), v * (0.98 + 0.03 * warm), v * (0.80 - 0.08 * warm)], 1)
    return col


def oak_albedo(P, front=False):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    across = y if front else z
    # long grain: streaks stretched along the planks, with slower drifting bands
    q = np.stack([x * 0.22, across * 7.0, 0 * x + (3.0 if front else 0.0)], 1)
    grain = fbm3(q, 1.0, 4, 3)
    bands = noise3(np.stack([x * 0.05, across * 1.3, 0 * x], 1), 1.0, 7)
    fine = noise3(np.stack([x * 1.2, across * 38.0, 0 * x], 1), 1.0, 9)
    a = OAK * (0.62 + 0.55 * grain)[:, None] * (0.86 + 0.28 * bands)[:, None] * (0.92 + 0.16 * fine)[:, None]
    # darker pores and old staining
    stain = smoothstep(0.58, 0.82, fbm3(np.stack([x * 0.3, across * 0.4, 0 * x + 5.0], 1), 1.0, 3, 13))[:, None]
    a = a * (1 - 0.35 * stain) + OAK_DARK * 0.35 * stain
    if not front:
        # the joint between two planks, and wear toward the front edge
        seam = np.exp(-((z - 5.6 - 0.12 * np.sin(x * 0.21)) / 0.045) ** 2)[:, None]
        a = a * (1 - 0.6 * seam)
        a = a * (1 + 0.18 * smoothstep(L.TABLE_FRONT - 1.2, L.TABLE_FRONT, z))[:, None]
    return a


def paint(sc, reserve=None, objects=()):
    """Wall first, then the table: broad lay-in, then smaller brushes where shadows fall."""
    cv, cam, lt, rng, px = sc.cv, sc.cam, sc.lt, sc.rng, sc.px
    W, H = sc.W, sc.H
    overlap = px(7)

    def free(xy):
        return np.ones(len(xy), bool) if reserve is None else ~reserve.inside(xy, overlap)

    # ---- wall
    wc = wall_colour(sc)
    horizon = cam.project(np.array([[0.0, 0.0, L.TABLE_BACK]]))[0, 1]
    ybot = min(H, horizon + px(40))
    big = Brush(kind="bristle", radius=px(46), nbristle=120, bristle_r=px(3.2), jitter=0.10, thick=1.3,
                pickup=0.45, aspect=0.45, angle=np.pi / 2)
    swirl = lambda xy: 0.55 + 0.8 * np.sin(xy[:, 0] / px(420) + 1.3 * np.sin(xy[:, 1] / px(300)))
    if ybot > 0:
        for k in range(2):
            fill_field(cv, (0, 0, W, ybot), wc, swirl, big, rng, length=px(240), spacing=px(24), curl=0.15, inside=free)
        fill_field(cv, (0, 0, W, ybot), wc, swirl, Brush(radius=px(26), hardness=0.25, thick=1.1, pickup=0.5), rng,
                   length=px(150), spacing=px(20), curl=0.2, inside=free)
        if reserve is not None and reserve.any():
            # cut in round the objects with smaller brushes, so the wall meets every edge
            outside = ~reserve
            x0, y0, x1, y1 = reserve.bbox(pad=int(px(110)))
            for r, band in ((12.0, 100.0), (5.0, 34.0)):
                near = lambda xy, band=band: free(xy) & (outside.depth(xy) < px(band))
                fill_field(cv, (x0, y0, x1, min(y1, ybot)), wc, swirl, Brush(radius=px(r), hardness=0.35, thick=1.2, pickup=0.5),
                           rng, length=px(r * 5), spacing=px(r * 0.8), curl=0.25, inside=near)
        fill_field(cv, (0, 0, W, ybot), wc, swirl, Brush(kind="blender", radius=px(34), hardness=0.15, pickup=0.35, load=0.5),
                   rng, length=px(200), spacing=px(30), curl=0.3, inside=free, paint=Paint.opaque("#000000"))
        cv.dry_now()

    # ---- table top
    top = forms.plane([0, 0, 0], [1.0, 0, 0], [0, 0, 1.0], flip=True)

    def top_col(P, N, u, v):
        ao = np.ones(len(P))
        for o in objects:
            ao = ao * o.ground_ao(P)
        return lt.shade(oak_albedo(P), P, N, ao=ao, spec=0.05, gloss=8)

    def vis(P, N, u, v):
        return free(cam.project(P))
    ur, vr = (-46.0, 46.0), (L.TABLE_BACK, L.TABLE_FRONT)
    grain_brush = Brush(kind="bristle", radius=px(20), nbristle=70, bristle_r=px(2.6), jitter=0.16, thick=1.3,
                        pickup=0.5, aspect=0.5, angle=np.pi / 2)
    paint_surface(cv, cam, top, top_col, grain_brush, rng, urange=ur, vrange=vr, length=px(170), spacing=px(13),
                  grid=90, visible=vis, coverage=1.3)
    paint_surface(cv, cam, top, top_col, grain_brush.but(radius=px(9), nbristle=36, bristle_r=px(1.8)), rng,
                  urange=ur, vrange=vr, length=px(90), spacing=px(7), grid=90, visible=vis)
    # shadows and their edges, with progressively smaller brushes
    for r, probe, tol in ((7.0, 16, 0.020), (3.2, 8, 0.022), (1.6, 4, 0.025)):
        paint_surface(cv, cam, top, top_col, Brush(radius=px(r), hardness=0.35, thick=1.1, pickup=0.5), rng,
                      urange=(-34, 40), vrange=(-28, L.TABLE_FRONT), length=px(r * 5), spacing=px(r * 0.8), grid=140,
                      visible=vis, detail=(tol, px(probe)))
    if reserve is not None and reserve.any():
        # and the table likewise, close in round every object
        outside = ~reserve
        near = lambda P, N, u, v: vis(P, N, u, v) & (outside.depth(cam.project(P)) < px(26))
        for r in (4.0, 1.8):
            paint_surface(cv, cam, top, top_col, Brush(radius=px(r), hardness=0.4, thick=1.1, pickup=0.5), rng,
                          urange=(-34, 40), vrange=(-28, L.TABLE_FRONT), length=px(r * 5), spacing=px(r * 0.8), grid=140,
                          visible=near)
    paint_surface(cv, cam, top, top_col, Brush(kind="blender", radius=px(5), hardness=0.2, pickup=0.3, load=0.5), rng,
                  urange=(-34, 40), vrange=(-28, L.TABLE_FRONT), length=px(40), spacing=px(9), grid=90, visible=vis,
                  paint=Paint.opaque("#000000"), detail=None)
    cv.dry_now()

    # ---- the table's front face and its worn edge
    front = forms.plane([0, 0, L.TABLE_FRONT], [1.0, 0, 0], [0, -1.0, 0], flip=True)

    def front_col(P, N, u, v):
        edge = smoothstep(0.5, 0.0, v)[:, None]              # the arris catches the light
        a = oak_albedo(P, front=True) * 0.78
        return lt.shade(a * (1 + 0.5 * edge), P, N, spec=0.04, gloss=8)
    paint_surface(cv, cam, front, front_col, grain_brush.but(radius=px(12), nbristle=44, bristle_r=px(2.0)), rng,
                  urange=ur, vrange=(0, L.TABLE_DROP), length=px(150), spacing=px(9), grid=60, visible=vis, coverage=1.4)
    paint_surface(cv, cam, front, front_col, Brush(radius=px(2.5), hardness=0.5, thick=1.1, pickup=0.5), rng,
                  urange=ur, vrange=(0, 0.9), length=px(40), spacing=px(2.4), grid=60, visible=vis)
    cv.dry_now()
