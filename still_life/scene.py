"""The set-up: where everything stands on the table, the viewpoint, and the window light."""
import numpy as np

from atelier.canvas import Canvas
from atelier.geom import Camera, Light, normalize
from atelier.shading import Lighting

W0, H0 = 3000, 2000          # the full canvas, in pixels

# --- the table (cm; x right, y up, z toward the viewer)
TABLE_FRONT = 17.0           # z of the front edge
TABLE_BACK = -30.0
TABLE_DROP = 9.0             # how much of the front face we may see

# --- where things stand
HOURGLASS = np.array([-5.8, 0.0, -6.0])
SKULL = np.array([2.6, 0.0, 8.2])
SKULL_YAW = np.radians(40.0)             # nose turned from "left" toward the viewer
SKULL_SCALE = 1.08                       # a big old tom
YARN = np.array([-15.0, 4.2, 2.5])
YARN_R = 4.2
COINS = np.array([-7.6, 0.0, 12.0])
BLADE = np.array([17.8, 0.0, 2.5])       # where the point enters the wood
BLADE_AXIS = normalize([-0.40, 0.91, -0.10])
ORANGE = np.array([11.2, 3.0, 6.0])
ORANGE_R = 3.0

LIGHT_DIR = (-0.45, 0.68, 0.58)          # toward the window: high, to the left and in front


class Scene:
    """Canvas, camera and light. `scale` shrinks everything for quick sketches;
    `crop` (in full-canvas pixels) paints only a window of the picture."""

    def __init__(self, scale=1.0, crop=None, seed=7, ground="#6a5236"):
        self.s = scale
        x0, y0, x1, y1 = crop if crop is not None else (0, 0, W0, H0)
        self.crop = (x0, y0, x1, y1)
        self.W = int(round((x1 - x0) * scale))
        self.H = int(round((y1 - y0) * scale))
        self.cam = Camera(eye=(1.6, 28.0, 80.0), target=(1.6, 5.0, 0.0), focal=5100.0 * scale,
                          W=self.W, H=self.H, cx=(W0 / 2 - x0) * scale, cy=(H0 / 2 + 100 - y0) * scale)
        self.light = Light(toward=LIGHT_DIR, soft=0.06)
        self.lt = Lighting(self.light, self.cam)
        self.cv = Canvas(self.W, self.H, ground=ground, seed=seed)
        self.rng = np.random.default_rng(seed)

    def px(self, v):
        """A size given in full-canvas pixels, at this scene's scale."""
        return v * self.s
