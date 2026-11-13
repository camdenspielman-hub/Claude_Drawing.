"""Ways of covering a form with strokes."""
import numpy as np

from .canvas import body_ks
from .geom import taper


class Batch:
    """Collects strokes for one brush and lays them in one go."""

    def __init__(self, canvas, brush, body=5.0):
        self.cv = canvas
        self.brush = brush
        self.body = body
        self.polys, self.K, self.S, self.press, self.rscale, self.tscale = [], [], [], [], [], []

    def add(self, poly, rgb=None, pressure=1.0, rscale=1.0, paint=None, tscale=1.0):
        poly = np.asarray(poly, dtype=np.float32)
        if len(poly) == 0:
            return
        if paint is None:
            K, S = body_ks(np.asarray(rgb, dtype=np.float64), self.body)
        else:
            K, S = paint.K, paint.S
        self.polys.append(poly)
        self.K.append(K)
        self.S.append(S)
        self.press.append(np.broadcast_to(np.asarray(pressure, np.float32), (len(poly),)))
        self.rscale.append(rscale)
        self.tscale.append(tscale)

    def flush(self, order="shuffle", rng=None):
        n = len(self.polys)
        if n == 0:
            return 0
        idx = np.arange(n)
        if order == "shuffle":
            (rng or np.random.default_rng(n)).shuffle(idx)
        elif isinstance(order, np.ndarray):
            idx = order
        self.cv.strokes([self.polys[i] for i in idx], np.array(self.K)[idx], np.array(self.S)[idx],
                        self.brush, [self.press[i] for i in idx], np.array(self.rscale)[idx],
                        np.array(self.tscale)[idx])
        self.polys, self.K, self.S, self.press, self.rscale, self.tscale = [], [], [], [], [], []
        return n


def _visible_run(mask):
    """Longest run of True that contains the middle of the stroke (start, stop) or None."""
    m = len(mask) // 2
    if not mask[m]:
        return None
    a = m
    while a > 0 and mask[a - 1]:
        a -= 1
    b = m
    while b < len(mask) - 1 and mask[b + 1]:
        b += 1
    return a, b + 1


def paint_surface(cv, cam, surf, color, brush, rng, *, length=40.0, spacing=6.0, along="u",
                  urange=(0.0, 1.0), vrange=(0.0, 1.0), npts=7, visible=None, body=5.0,
                  coverage=1.0, wobble=0.25, head=0.25, tail=0.3, press=(0.25, 1.0), rjit=0.0,
                  order="shuffle", paint=None, grid=28, slant=0.0, min_facing=0.02,
                  stencil=None, margin=None, detail=None, thick_by_value=None, broken=0.0, recolour=None):
    """Cover a parametric surface with strokes that follow one of its parameter directions.

    surf(u, v)            -> points P (..., 3) and outward normals N (..., 3)
    color(P, N, u, v)     -> sRGB colour per stroke (M, 3), asked at the middle of each stroke
    visible(P, N, u, v)   -> optional boolean mask (hidden parts are simply not painted)
    length, spacing       -> stroke length and the gap between neighbouring strokes, in pixels
    stencil, margin       -> keep the brush inside a drawn outline (margin defaults to its radius)
    thick_by_value=(a, b) -> paint thin (a) in the darks and thick (b) in the lights, as in oil
                             painting, so darks stay transparent and lights have body
    broken                -> how much each stroke's mixture strays from the exact colour (broken colour)
    recolour(rgb)         -> last say on each stroke's colour (e.g. turn it to a monochrome underpainting)
    detail=(tol, probe)   -> only place strokes where the colour changes by more than `tol`
                             within `probe` px: a smaller brush going back in where it is needed.
                             (tol, probe, "calm") does the reverse: only where it does not change,
                             which is where a blending brush may safely go
    """
    u0, u1 = urange
    v0, v1 = vrange
    # --- measure the surface on the canvas so strokes come out the right size everywhere
    gu = np.linspace(u0, u1, grid + 1)
    gv = np.linspace(v0, v1, grid + 1)
    U, Vv = np.meshgrid(gu, gv, indexing="ij")
    P, N = surf(U, Vv)
    xy = cam.project(P)
    du = np.linalg.norm(xy[1:, :-1] - xy[:-1, :-1], axis=-1) / (gu[1] - gu[0])   # px per unit u
    dv = np.linalg.norm(xy[:-1, 1:] - xy[:-1, :-1], axis=-1) / (gv[1] - gv[0])   # px per unit v
    a = xy[1:, :-1] - xy[:-1, :-1]
    b = xy[:-1, 1:] - xy[:-1, :-1]
    area = np.abs(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0])                  # px^2 per cell
    Pc = 0.25 * (P[1:, 1:] + P[:-1, :-1] + P[1:, :-1] + P[:-1, 1:])
    Nc = N[:-1, :-1] + N[1:, 1:] + N[1:, :-1] + N[:-1, 1:]
    Nc = Nc / np.maximum(np.linalg.norm(Nc, axis=-1, keepdims=True), 1e-9)
    facing = np.sum(Nc * cam.to_eye(Pc), axis=-1)
    area = np.where(facing > -0.15, area, 0.0)
    # only the part of the surface that falls on the canvas (and inside its outline) needs strokes
    bx0, by0, bx1, by1 = 0.0, 0.0, float(cv.W), float(cv.H)
    if stencil is not None:
        sx0, sy0, sx1, sy1 = stencil.bbox(pad=int(length))
        bx0, by0, bx1, by1 = max(bx0, sx0), max(by0, sy0), min(bx1, sx1), min(by1, sy1)
    cx0 = np.minimum(np.minimum(xy[:-1, :-1, 0], xy[1:, :-1, 0]), np.minimum(xy[:-1, 1:, 0], xy[1:, 1:, 0]))
    cx1 = np.maximum(np.maximum(xy[:-1, :-1, 0], xy[1:, :-1, 0]), np.maximum(xy[:-1, 1:, 0], xy[1:, 1:, 0]))
    cy0 = np.minimum(np.minimum(xy[:-1, :-1, 1], xy[1:, :-1, 1]), np.minimum(xy[:-1, 1:, 1], xy[1:, 1:, 1]))
    cy1 = np.maximum(np.maximum(xy[:-1, :-1, 1], xy[1:, :-1, 1]), np.maximum(xy[:-1, 1:, 1], xy[1:, 1:, 1]))
    ox = np.clip(np.minimum(cx1, bx1) - np.maximum(cx0, bx0), 0, None)
    oy = np.clip(np.minimum(cy1, by1) - np.maximum(cy0, by0), 0, None)
    share = (ox * oy) / np.maximum((cx1 - cx0) * (cy1 - cy0), 1e-9)
    area = area * np.clip(share * 1.15, 0, 1)
    want = area / (0.62 * length * spacing) * coverage
    total = want.sum()
    if total < 0.5:
        return 0
    M = int(rng.poisson(total)) if total < 50 else int(round(total))
    if M == 0:
        return 0
    cell = rng.choice(want.size, size=M, p=(want / total).ravel())
    ci, cj = np.unravel_index(cell, want.shape)
    us = gu[ci] + rng.random(M) * (gu[1] - gu[0])
    vs = gv[cj] + rng.random(M) * (gv[1] - gv[0])
    if stencil is not None or True:
        # drop seeds that land off the canvas before doing any more work on them
        sxy = cam.project(surf(us, vs)[0])
        keep = (sxy[:, 0] > bx0 - length) & (sxy[:, 0] < bx1 + length) & (sxy[:, 1] > by0 - length) & (sxy[:, 1] < by1 + length)
        us, vs, ci, cj = us[keep], vs[keep], ci[keep], cj[keep]
        M = len(us)
        if M == 0:
            return 0
    su = np.maximum(du[ci, cj], 1e-6)
    sv = np.maximum(dv[ci, cj], 1e-6)

    if detail is not None and color is not None:
        tol, probe = detail[0], detail[1]
        calm = len(detail) > 2 and detail[2] == "calm"      # the opposite: only where little changes
        c0 = color(*surf(us, vs), us, vs)
        dmax = np.zeros(M)
        for du_, dv_ in ((probe / su, 0), (-probe / su, 0), (0, probe / sv), (0, -probe / sv)):
            uq, vq = np.clip(us + du_, u0, u1), np.clip(vs + dv_, v0, v1)
            dmax = np.maximum(dmax, np.abs(color(*surf(uq, vq), uq, vq) - c0).max(axis=1))
        keep = (dmax <= tol) if calm else (dmax > tol)
        us, vs, su, sv = us[keep], vs[keep], su[keep], sv[keep]
        M = len(us)
        if M == 0:
            return 0

    k = np.linspace(-0.5, 0.5, npts)[None, :]
    drift = rng.normal(0, 1, (M, 1)) * wobble * spacing * k * 2 + slant * length * k
    if along == "u":
        Up = us[:, None] + k * (length / su)[:, None]
        Vp = vs[:, None] + drift / sv[:, None]
    else:
        Vp = vs[:, None] + k * (length / sv)[:, None]
        Up = us[:, None] + drift / su[:, None]
    inside = (Up >= u0) & (Up <= u1) & (Vp >= v0) & (Vp <= v1)
    Up = np.clip(Up, u0, u1)
    Vp = np.clip(Vp, v0, v1)
    Pp, Np = surf(Up, Vp)
    xyp = cam.project(Pp)
    ok = inside & (np.sum(Np * cam.to_eye(Pp), axis=-1) > min_facing)
    if visible is not None:
        ok &= visible(Pp, Np, Up, Vp)
    if stencil is not None:
        ok &= stencil.inside(xyp, brush.radius * 0.85 if margin is None else margin)

    mid = npts // 2
    cols = color(Pp[:, mid], Np[:, mid], Up[:, mid], Vp[:, mid]) if paint is None else None
    ts = np.ones(M)
    if cols is not None:
        if recolour is not None:
            cols = recolour(cols)
        if broken > 0:
            # no two brush-loads are mixed quite alike: a little lighter or darker, warmer or cooler
            k = 1 + broken * rng.normal(0, 1, (M, 1))
            warm = broken * 0.6 * rng.normal(0, 1, (M, 1))
            cols = np.clip(cols * k * (1 + warm * np.array([1.0, 0.0, -1.0])), 0, 1)
        if thick_by_value is not None:
            y = 0.30 * cols[:, 0] + 0.59 * cols[:, 1] + 0.11 * cols[:, 2]
            t = np.clip((y - 0.06) / 0.40, 0, 1)
            ts = thick_by_value[0] + (thick_by_value[1] - thick_by_value[0]) * t * t * (3 - 2 * t)
    batch = Batch(cv, brush, body)
    for i in range(M):
        run = _visible_run(ok[i])
        if run is None or run[1] - run[0] < 2:
            continue
        a_, b_ = run
        n = b_ - a_
        pr = taper(n, head, tail, press[0], press[1])
        rs = 1.0 + rjit * rng.normal()
        if paint is None:
            batch.add(xyp[i, a_:b_], cols[i], pr, max(0.4, rs), tscale=ts[i])
        else:
            batch.add(xyp[i, a_:b_], None, pr, max(0.4, rs), paint=paint)
    return batch.flush(order=order, rng=rng)


def paint_solid(cv, cam, surf, color, brush, rng, *, region, flow, which="u", length=40.0, spacing=6.0,
                npts=7, coverage=1.0, wobble=0.25, bend=0.2, head=0.25, tail=0.3, press=(0.25, 1.0), rjit=0.0,
                order="shuffle", paint=None, margin=None, detail=None, thick_by_value=None, broken=0.0,
                recolour=None, visible=None, body=5.0):
    """Cover what shows of a sculpted solid with strokes that follow its form.

    surf(x, y)            -> the point of the solid under canvas spot (x, y) and its normal (zero if none)
    region                -> the part of the canvas this form shows in (a Stencil)
    flow(P, N, which)     -> the direction (in space) a stroke should run at P: which="u" one way
                             over the form, "v" across it
    The other arguments are as for paint_surface.
    """
    A = region.area()
    M = int(round(A / (0.62 * length * spacing) * coverage))
    if M <= 0:
        return 0
    ys, xs = np.nonzero(region.mask)
    pick = rng.integers(0, len(xs), M)
    seeds = np.stack([xs[pick] + rng.random(M), ys[pick] + rng.random(M)], axis=1)
    P, N = surf(seeds[:, 0], seeds[:, 1])
    keep = np.any(N != 0, axis=1)
    if visible is not None:
        keep &= visible(P, N, seeds[:, 0], seeds[:, 1])
    seeds, P, N = seeds[keep], P[keep], N[keep]
    M = len(seeds)
    if M == 0:
        return 0

    if detail is not None and color is not None:
        tol, probe = detail[0], detail[1]
        calm = len(detail) > 2 and detail[2] == "calm"
        c0 = color(P, N, seeds[:, 0], seeds[:, 1])
        dmax = np.zeros(M)
        for dx, dy in ((probe, 0), (-probe, 0), (0, probe), (0, -probe)):
            q = seeds + np.array([dx, dy])
            Pq, Nq = surf(q[:, 0], q[:, 1])
            there = np.any(Nq != 0, axis=1)
            if there.any():
                cq = color(Pq[there], Nq[there], q[there, 0], q[there, 1])
                dmax[there] = np.maximum(dmax[there], np.abs(cq - c0[there]).max(axis=1))
        keep = (dmax <= tol) if calm else (dmax > tol)
        seeds, P, N = seeds[keep], P[keep], N[keep]
        M = len(seeds)
        if M == 0:
            return 0

    d3 = flow(P, N, which)
    d2 = cam.project(P + d3 * 0.05) - cam.project(P)
    d2 = d2 / np.maximum(np.linalg.norm(d2, axis=1, keepdims=True), 1e-9)
    nr = np.stack([-d2[:, 1], d2[:, 0]], axis=1)
    k = np.linspace(-0.5, 0.5, npts)[None, :]
    side = (rng.normal(0, 1, (M, 1)) * wobble * spacing * k * 2
            + bend * rng.normal(0, 1, (M, 1)) * length * (k * k - 1.0 / 12.0))
    xyp = seeds[:, None, :] + (k * length)[..., None] * d2[:, None, :] + side[..., None] * nr[:, None, :]
    ok = region.inside(xyp.reshape(-1, 2), brush.radius * 0.85 if margin is None else margin).reshape(M, npts)

    cols = color(P, N, seeds[:, 0], seeds[:, 1]) if paint is None else None
    return _lay_strokes(cv, brush, body, xyp, ok, cols, paint, rng, head, tail, press, rjit, order,
                        thick_by_value, broken, recolour)


def _lay_strokes(cv, brush, body, xyp, ok, cols, paint, rng, head, tail, press, rjit, order,
                 thick_by_value, broken, recolour):
    M = len(xyp)
    ts = np.ones(M)
    if cols is not None:
        if recolour is not None:
            cols = recolour(cols)
        if broken > 0:
            k = 1 + broken * rng.normal(0, 1, (M, 1))
            warm = broken * 0.6 * rng.normal(0, 1, (M, 1))
            cols = np.clip(cols * k * (1 + warm * np.array([1.0, 0.0, -1.0])), 0, 1)
        if thick_by_value is not None:
            y = 0.30 * cols[:, 0] + 0.59 * cols[:, 1] + 0.11 * cols[:, 2]
            t = np.clip((y - 0.06) / 0.40, 0, 1)
            ts = thick_by_value[0] + (thick_by_value[1] - thick_by_value[0]) * t * t * (3 - 2 * t)
    batch = Batch(cv, brush, body)
    for i in range(M):
        run = _visible_run(ok[i])
        if run is None or run[1] - run[0] < 2:
            continue
        a_, b_ = run
        pr = taper(b_ - a_, head, tail, press[0], press[1])
        rs = 1.0 + rjit * rng.normal()
        if paint is None:
            batch.add(xyp[i, a_:b_], cols[i], pr, max(0.4, rs), tscale=ts[i])
        else:
            batch.add(xyp[i, a_:b_], None, pr, max(0.4, rs), paint=paint)
    return batch.flush(order=order, rng=rng)


def fill_field(cv, bbox, color, angle, brush, rng, *, length=40.0, spacing=6.0, inside=None, npts=6,
               body=5.0, coverage=1.0, head=0.25, tail=0.3, press=(0.25, 1.0), rjit=0.0, paint=None,
               curl=0.0, order="shuffle", stencil=None, margin=None):
    """Cover a region of the canvas itself with strokes that follow a direction field.

    color(xy) -> sRGB (M, 3); angle(xy) -> stroke direction in radians (M,); inside(xy) -> bool.
    """
    x0, y0, x1, y1 = bbox
    M = int((x1 - x0) * (y1 - y0) / (0.62 * length * spacing) * coverage)
    if M <= 0:
        return 0
    seeds = np.stack([x0 + rng.random(M) * (x1 - x0), y0 + rng.random(M) * (y1 - y0)], axis=1)
    step = length / (npts - 1)
    pts = np.empty((M, npts, 2))
    mid = npts // 2
    pts[:, mid] = seeds
    for direction, rng_k in ((1, range(mid + 1, npts)), (-1, range(mid - 1, -1, -1))):
        cur = seeds.copy()
        for kk in rng_k:
            ang = angle(cur) + curl * rng.normal(0, 1, M)
            cur = cur + direction * step * np.stack([np.cos(ang), np.sin(ang)], axis=1)
            pts[:, kk] = cur
    ok = np.ones((M, npts), bool) if inside is None else inside(pts.reshape(-1, 2)).reshape(M, npts)
    if stencil is not None:
        ok &= stencil.inside(pts, brush.radius * 0.85 if margin is None else margin)
    cols = color(seeds) if paint is None else None
    batch = Batch(cv, brush, body)
    for i in range(M):
        run = _visible_run(ok[i])
        if run is None or run[1] - run[0] < 2:
            continue
        a_, b_ = run
        pr = taper(b_ - a_, head, tail, press[0], press[1])
        rs = max(0.4, 1.0 + rjit * rng.normal())
        if paint is None:
            batch.add(pts[i, a_:b_], cols[i], pr, rs)
        else:
            batch.add(pts[i, a_:b_], None, pr, rs, paint=paint)
    return batch.flush(order=order, rng=rng)


def line(cv, pts, paint, brush, pressure=None, head=0.15, tail=0.25, lo=0.2, hi=1.0):
    """A single drawn line through canvas points."""
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 2:
        return
    if pressure is None:
        pressure = taper(len(pts), head, tail, lo, hi)
    cv.stroke(pts, paint, brush, pressure)


def paint_form(sc, surf, ur, vr, col, st, *, r1=5.0, r2=2.3, along="u", vis=None, grid=40, blend=3.0,
               edge=True, detail=True, coverage=1.2, body=5.0, r_detail=1.2, tol=0.03):
    """Paint one solid form the way the whole picture is painted:
    lay-in with a larger brush, modelling across it with a smaller one, a soft brush to melt
    the strokes, sharp touches where the light changes quickly, then the edge with a fine point.
    Sizes are in full-canvas pixels; `st` is the form's drawn outline.
    """
    from .canvas import Brush, Paint
    cv, cam, rng, px = sc.cv, sc.cam, sc.rng, sc.px
    other = "u" if along == "v" else "v"
    n = 0
    n += paint_surface(cv, cam, surf, col, Brush(radius=px(r1), hardness=0.5, thick=1.4, pickup=0.5), rng,
                       urange=ur, vrange=vr, along=along, length=px(r1 * 4.5), spacing=px(r1 * 0.7), stencil=st,
                       visible=vis, grid=grid, coverage=coverage, body=body)
    if r2:
        n += paint_surface(cv, cam, surf, col, Brush(radius=px(r2), hardness=0.55, thick=1.2, pickup=0.55), rng,
                           urange=ur, vrange=vr, along=other, length=px(r2 * 5), spacing=px(r2 * 0.75), stencil=st,
                           margin=px(r2 * 0.5), visible=vis, grid=grid, body=body)
    if detail and r2:
        # wherever light gives way to shadow within a brush-length, go back in with short touches,
        # so the edge of a shadow is drawn rather than left as the ragged ends of longer strokes
        r_mid = max(r_detail, 0.75 * r2)
        n += paint_surface(cv, cam, surf, col, Brush(radius=px(r_mid), hardness=0.5, thick=1.2, pickup=0.6), rng,
                           urange=ur, vrange=vr, along=other, length=px(r_mid * 3.0), spacing=px(r_mid * 0.8),
                           stencil=st, margin=px(r_mid * 0.5), visible=vis, grid=grid, detail=(tol, px(11.0)), body=body)
    if blend:
        # melt the strokes together, but only in the quiet passages: a shadow's edge is left as drawn
        paint_surface(cv, cam, surf, col, Brush(kind="blender", radius=px(blend), hardness=0.2, pickup=0.30, load=0.5),
                      rng, urange=ur, vrange=vr, along=along, length=px(10), spacing=px(4.0), stencil=st, visible=vis,
                      grid=grid, paint=Paint.opaque("#000000"), detail=(tol * 0.8, px(10.0), "calm"))
    if detail:
        n += paint_surface(cv, cam, surf, col, Brush(radius=px(r_detail), hardness=0.55, thick=1.2, pickup=0.6), rng,
                           urange=ur, vrange=vr, along=along, length=px(r_detail * 3.2), spacing=px(r_detail * 0.8),
                           stencil=st, margin=px(r_detail * 0.5), visible=vis, grid=grid, detail=(tol, px(3.5)), body=body)
    if edge:
        def near_edge(P, N, u, v):
            ok = st.depth(cam.project(P)) < px(3.2)
            return ok if vis is None else ok & vis(P, N, u, v)
        n += paint_surface(cv, cam, surf, col, Brush(radius=px(1.0), hardness=0.7, thick=1.2, pickup=0.6), rng,
                           urange=ur, vrange=vr, along=along, length=px(6), spacing=px(1.1), stencil=st, margin=px(0.5),
                           visible=near_edge, grid=grid, body=body)
    return n
