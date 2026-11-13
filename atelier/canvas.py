"""Python side of the paint engine: canvas, paints and brushes."""
import ctypes
import os

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_HERE, "..", "engine", "libpaint.so"))

_fp = ctypes.POINTER(ctypes.c_float)
_ip = ctypes.POINTER(ctypes.c_int)


class _Canvas(ctypes.Structure):
    _fields_ = [("W", ctypes.c_int), ("H", ctypes.c_int), ("dry", _fp), ("K", _fp), ("S", _fp),
                ("V", _fp), ("hgt", _fp), ("tooth", _fp)]


class _Brush(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("radius", ctypes.c_float), ("hardness", ctypes.c_float),
                ("aspect", ctypes.c_float), ("angle", ctypes.c_float), ("follow", ctypes.c_int),
                ("spacing", ctypes.c_float), ("thick", ctypes.c_float), ("pickup", ctypes.c_float),
                ("load", ctypes.c_float), ("nbristle", ctypes.c_int), ("bristle_r", ctypes.c_float),
                ("jitter", ctypes.c_float), ("tooth", ctypes.c_float),
                ("press_radius", ctypes.c_float), ("press_thick", ctypes.c_float),
                ("seed", ctypes.c_uint32)]


_lib.stroke.argtypes = [ctypes.POINTER(_Canvas), ctypes.POINTER(_Brush), _fp, _fp, ctypes.c_int, _fp, _fp, _fp]
_lib.strokes.argtypes = [ctypes.POINTER(_Canvas), ctypes.POINTER(_Brush), ctypes.c_int, _ip,
                         _fp, _fp, _fp, _fp, _fp, _fp]
_lib.dry_canvas.argtypes = [ctypes.POINTER(_Canvas)]
_lib.render.argtypes = [ctypes.POINTER(_Canvas), _fp, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                        ctypes.c_float]


_lib.preview.argtypes = [ctypes.POINTER(_Canvas), _fp, ctypes.c_int]
_lib.edt.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, _fp]


def distance_inside(mask):
    """Distance in px from each True pixel of `mask` to the nearest False pixel."""
    m = np.ascontiguousarray(mask, dtype=np.uint8)
    out = np.empty(m.shape, np.float32)
    _lib.edt(m.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), m.shape[1], m.shape[0], _ptr(out))
    return out


def _ptr(a):
    return a.ctypes.data_as(_fp)


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def hexrgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def as_rgb(c):
    """Accept '#rrggbb' or an (r, g, b) triple in 0..1."""
    return hexrgb(c) if isinstance(c, str) else np.asarray(c, dtype=np.float64)


# Scattering is a little stronger toward blue, as it is for real white pigment.
# This is what makes a thin light scumble over a dark ground turn cool.
_S_TINT = np.array([0.92, 1.0, 1.10])


def body_ks(rgb, body=5.0):
    """K and S of an opaque body colour whose thick-film colour is `rgb` (sRGB 0..1).

    Works on a single colour or an (N, 3) array.
    """
    R = np.clip(srgb_to_linear(rgb), 0.003, 0.965)
    S = body * _S_TINT * np.ones_like(R)
    K = S * (1 - R) ** 2 / (2 * R)
    return K.astype(np.float32), S.astype(np.float32)


class Paint:
    """A mixed pile of paint: absorption K and scattering S per unit volume."""

    def __init__(self, K, S):
        self.K = np.ascontiguousarray(K, dtype=np.float32)
        self.S = np.ascontiguousarray(S, dtype=np.float32)

    @staticmethod
    def opaque(rgb, body=5.0):
        K, S = body_ks(as_rgb(rgb), body)
        return Paint(K, S)

    @staticmethod
    def glaze(rgb, density=1.0):
        """A transparent colour: a unit film over white looks like `rgb`."""
        T = np.clip(srgb_to_linear(as_rgb(rgb)), 0.002, 0.999)
        return Paint(-np.log(T) / 2 * density, np.full(3, 0.02))

    def mix(self, other, t):
        return Paint(self.K * (1 - t) + other.K * t, self.S * (1 - t) + other.S * t)

    def thin(self, f):
        """Dilute with medium: the same film is `f` times as strong."""
        return Paint(self.K * f, self.S * f)


class Brush:
    """Brush settings. See engine/paint.cpp for what each one does."""

    def __init__(self, radius=8.0, kind="soft", hardness=0.6, aspect=1.0, angle=0.0, follow=True,
                 spacing=0.25, thick=1.0, pickup=0.5, load=0.0, nbristle=40, bristle_r=1.0,
                 jitter=0.0, tooth=0.0, press_radius=0.5, press_thick=0.5):
        self.kind = kind
        self.radius = radius
        self.hardness = hardness
        self.aspect = aspect
        self.angle = angle
        self.follow = follow
        self.spacing = spacing
        self.thick = thick
        self.pickup = pickup
        self.load = load
        self.nbristle = nbristle
        self.bristle_r = bristle_r
        self.jitter = jitter
        self.tooth = tooth
        self.press_radius = press_radius
        self.press_thick = press_thick

    def but(self, **kw):
        """A copy with some settings changed."""
        b = Brush.__new__(Brush)
        b.__dict__.update(self.__dict__)
        b.__dict__.update(kw)
        return b

    def _c(self, seed):
        return _Brush({"soft": 0, "bristle": 1, "blender": 2}[self.kind], self.radius, self.hardness, self.aspect,
                      self.angle, 1 if self.follow else 0, self.spacing, self.thick, self.pickup,
                      self.load, int(self.nbristle), self.bristle_r, self.jitter, self.tooth,
                      self.press_radius, self.press_thick, int(seed) & 0xFFFFFFFF)


def value_noise(W, H, scale, rng, aniso=1.0):
    """Smooth noise in 0..1 with features about `scale` px across (`aniso` stretches them in x)."""
    w = max(2, int(W / (scale * aniso)) + 2)
    h = max(2, int(H / scale) + 2)
    g = (rng.random((h, w)) * 255).astype(np.uint8)
    im = Image.fromarray(g).resize((W, H), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def panel_tooth(W, H, seed=1):
    """Ground texture of a primed panel: brushed priming streaks plus fine grain."""
    rng = np.random.default_rng(seed)
    t = 0.45 * value_noise(W, H, 2.2, rng)
    t += 0.30 * value_noise(W, H, 3.0, rng, aniso=14.0)
    t += 0.25 * value_noise(W, H, 9.0, rng)
    t = (t - t.min()) / (t.max() - t.min())
    return np.ascontiguousarray(t, dtype=np.float32)


class Recorder:
    """Saves a small frame of the canvas every so much brushwork, for the time-lapse.

    Pacing is by painted area (stroke length x width), not stroke count, so the film
    moves at an even speed through broad lay-ins and fine finishing alike.
    """

    def __init__(self, folder, area_per_frame, stride=3, quality=90):
        os.makedirs(folder, exist_ok=True)
        self.folder = folder
        self.area_per_frame = float(area_per_frame)
        self.stride = stride
        self.quality = quality
        self.acc = 0.0
        self.n = 0

    def snap(self, canvas):
        ow = (canvas.W + self.stride - 1) // self.stride
        oh = (canvas.H + self.stride - 1) // self.stride
        out = np.empty((oh, ow, 3), np.float32)
        _lib.preview(ctypes.byref(canvas._c), _ptr(out), self.stride)
        Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
            os.path.join(self.folder, "f%05d.jpg" % self.n), quality=self.quality)
        self.n += 1


class Canvas:
    def __init__(self, W, H, ground="#8a6f4e", seed=1):
        self.W, self.H = W, H
        g = srgb_to_linear(as_rgb(ground)).astype(np.float32)
        self.dry = np.ascontiguousarray(np.broadcast_to(g, (H, W, 3)), dtype=np.float32).copy()
        self.K = np.zeros((H, W, 3), np.float32)
        self.S = np.zeros((H, W, 3), np.float32)
        self.V = np.zeros((H, W), np.float32)
        self.hgt = np.zeros((H, W), np.float32)
        self.tooth = panel_tooth(W, H, seed)
        self._c = _Canvas(W, H, _ptr(self.dry), _ptr(self.K), _ptr(self.S), _ptr(self.V),
                          _ptr(self.hgt), _ptr(self.tooth))
        self._seed = 1000 * seed
        self.nstrokes = 0
        self.recorder = None

    def _next_seed(self):
        self._seed += 1
        return self._seed

    def stroke(self, pts, paint, brush, pressure=1.0):
        """One stroke through `pts` (N x 2). `pressure` is a number or one value per point."""
        pts = np.asarray(pts, dtype=np.float32)
        n = len(pts)
        xs = np.ascontiguousarray(pts[:, 0])
        ys = np.ascontiguousarray(pts[:, 1])
        ps = np.ascontiguousarray(np.broadcast_to(np.asarray(pressure, np.float32), (n,)))
        b = brush._c(self._next_seed())
        _lib.stroke(ctypes.byref(self._c), ctypes.byref(b), _ptr(paint.K), _ptr(paint.S), n,
                    _ptr(xs), _ptr(ys), _ptr(ps))
        self.nstrokes += 1
        rec = self.recorder
        if rec is not None and n > 1:
            rec.acc += float(np.hypot(np.diff(xs), np.diff(ys)).sum() + brush.radius) * 2 * brush.radius
            if rec.acc >= rec.area_per_frame:
                rec.acc = 0.0
                rec.snap(self)

    def strokes(self, polys, K, S, brush, pressures=None, rscale=None):
        """Many strokes with one brush. polys: list of (N, 2); K, S: (M, 3) paint per stroke."""
        m = len(polys)
        if m == 0:
            return
        counts = np.array([len(p) for p in polys], dtype=np.int32)
        allp = np.concatenate(polys).astype(np.float32)
        xs = np.ascontiguousarray(allp[:, 0])
        ys = np.ascontiguousarray(allp[:, 1])
        if pressures is None:
            ps = np.ones(len(allp), np.float32)
        else:
            ps = np.ascontiguousarray(np.concatenate(
                [np.broadcast_to(np.asarray(q, np.float32), (len(p),)) for p, q in zip(polys, pressures)]))
        K = np.ascontiguousarray(K, dtype=np.float32).reshape(m, 3)
        S = np.ascontiguousarray(S, dtype=np.float32).reshape(m, 3)
        rs = np.ascontiguousarray(np.ones(m, np.float32) if rscale is None
                                  else np.asarray(rscale, np.float32))
        b = brush._c(self._next_seed())
        self._seed += m
        rec = self.recorder
        if rec is None:
            cuts = [0, m]
        else:
            # split the batch wherever the time-lapse is due a frame
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            seg = np.hypot(np.diff(xs, append=xs[-1:]), np.diff(ys, append=ys[-1:]))
            seg[starts[1:] - 1] = 0.0
            length = np.add.reduceat(seg, starts)
            area = (length + brush.radius) * 2 * brush.radius * rs
            total = rec.acc + np.cumsum(area)
            frame_no = np.floor(total / rec.area_per_frame).astype(np.int64)
            cuts = [0] + list(np.nonzero(np.diff(frame_no, prepend=0) > 0)[0] + 1) + [m]
            rec.acc = float(total[-1] - frame_no[-1] * rec.area_per_frame)
        off = np.concatenate([[0], np.cumsum(counts)])
        seed0 = b.seed
        for a_, b_ in zip(cuts[:-1], cuts[1:]):
            if b_ > a_:
                b.seed = (seed0 + 7919 * int(a_)) & 0xFFFFFFFF
                _lib.strokes(ctypes.byref(self._c), ctypes.byref(b), int(b_ - a_),
                             counts[a_:b_].ctypes.data_as(_ip),
                             _ptr(xs[off[a_]:]), _ptr(ys[off[a_]:]), _ptr(ps[off[a_]:]),
                             _ptr(K[a_:]), _ptr(S[a_:]), _ptr(rs[a_:]))
            if rec is not None and b_ < m:
                rec.snap(self)
        self.nstrokes += m

    def dry_now(self):
        _lib.dry_canvas(ctypes.byref(self._c))

    def array(self, relief=0.0, light=(-0.6, -0.8), tooth_amt=0.0):
        out = np.empty((self.H, self.W, 3), np.float32)
        _lib.render(ctypes.byref(self._c), _ptr(out), relief, light[0], light[1], tooth_amt)
        return out

    def image(self, relief=0.0, tooth_amt=0.0):
        a = self.array(relief=relief, tooth_amt=tooth_amt)
        return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))

    def save(self, path, relief=0.0, tooth_amt=0.0, scale=None, crop=None):
        im = self.image(relief=relief, tooth_amt=tooth_amt)
        if crop is not None:
            im = im.crop(crop)
        if scale is not None and scale != 1:
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
        im.save(path)
        return path
