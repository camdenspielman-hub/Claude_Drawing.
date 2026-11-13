"""The set-up: where everything stands on the table, the viewpoint, and the window light.

The arrangement is the aftermath of a cat's play, long after the cat: the hourglass knocked on
its side, coins scattered, the yarn dragged across everything and still caught in the skull's
teeth, and the strand cut where a blade has been driven into the table.
"""
import numpy as np

from atelier.canvas import Canvas
from atelier.geom import Camera, Light, normalize
from atelier.shading import Lighting

W0, H0 = 3000, 2000          # the full canvas, in pixels

# --- the table and the wall behind it (cm; x right, y up, z toward the viewer)
TABLE_FRONT = 17.0           # z of the front edge
TABLE_BACK = -24.0           # where the table meets the wall
WALL_Z = -24.0
TABLE_DROP = 9.0             # how much of the front face we may see

# --- where things stand
YARN = np.array([-13.4, 3.75, 6.6])
YARN_R = 4.0
HOURGLASS = np.array([7.4, 4.6, -4.4])             # centre of the end board nearest us (it lies on its side)
HOURGLASS_AXIS = normalize([-0.951, 0.0, -0.309])  # from that end to the far one
SKULL = np.array([3.4, 0.0, 10.2])
SKULL_YAW = np.radians(30.0)                       # nose turned from "left" toward the viewer
SKULL_SCALE = 0.95                                 # a grown cat: nine and a half centimetres
COINS = np.array([-5.0, 0.0, 10.2])
ORANGE = np.array([12.9, 3.8, 0.6])
ORANGE_R = 3.8
BLADE = np.array([18.4, 0.0, 5.6])                 # where the point enters the wood
BLADE_AXIS = normalize([-0.38, 0.91, -0.16])

LIGHT_DIR = (-0.80, 0.56, 0.20)                    # toward the window: on the left, raking across the table
EYE = (0.6, 26.0, 74.0)
LOOK_AT = (0.6, 5.6, 0.0)
FOCAL = 4700.0


class Scene:
    """Canvas, camera and light. `scale` shrinks everything for quick sketches;
    `crop` (in full-canvas pixels) paints only a window of the picture."""

    def __init__(self, scale=1.0, crop=None, seed=7, ground="#6a5236"):
        self.s = scale
        x0, y0, x1, y1 = crop if crop is not None else (0, 0, W0, H0)
        self.crop = (x0, y0, x1, y1)
        self.W = int(round((x1 - x0) * scale))
        self.H = int(round((y1 - y0) * scale))
        self.cam = Camera(eye=EYE, target=LOOK_AT, focal=FOCAL * scale, W=self.W, H=self.H,
                          cx=(W0 / 2 - x0) * scale, cy=(H0 / 2 + 130 - y0) * scale)
        self.light = Light(toward=LIGHT_DIR, soft=0.20)
        self.lt = Lighting(self.light, self.cam, exposure=1.5)
        self.lt.wall_z = WALL_Z
        self.cv = Canvas(self.W, self.H, ground=ground, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.objs = {}

    def px(self, v):
        """A size given in full-canvas pixels, at this scene's scale."""
        return v * self.s

    def full(self, xy):
        """Canvas points -> where they are on the full, unscaled canvas."""
        return np.asarray(xy) / self.s + np.array([self.crop[0], self.crop[1]])
