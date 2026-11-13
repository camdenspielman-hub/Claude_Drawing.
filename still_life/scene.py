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
HOURGLASS = np.array([-4.5, 0.0, -7.5])
SKULL = np.array([6.2, 0.0, 3.2])
SKULL_YAW = np.radians(38.0)             # nose turned from "left" toward the viewer
YARN = np.array([-14.2, 4.2, 2.0])
YARN_R = 4.2
COINS = np.array([-4.5, 0.0, 10.5])
BLADE = np.array([16.8, 0.0, 2.0])       # where the point enters the wood
BLADE_AXIS = normalize([-0.38, 0.92, -0.10])
ORANGE = np.array([12.4, 3.3, 10.6])
ORANGE_R = 3.3

LIGHT_DIR = (-0.55, 0.70, 0.45)          # toward the window: high, left, a little in front


class Scene:
    """Canvas, camera and light. `scale` shrinks everything for quick sketches;
    `crop` (in full-canvas pixels) paints only a window of the picture."""

    def __init__(self, scale=1.0, crop=None, seed=7, ground="#6a5236"):
        self.s = scale
        x0, y0, x1, y1 = crop if crop is not None else (0, 0, W0, H0)
        self.crop = (x0, y0, x1, y1)
        self.W = int(round((x1 - x0) * scale))
        self.H = int(round((y1 - y0) * scale))
        self.cam = Camera(eye=(0.3, 28.0, 80.0), target=(0.3, 5.0, 0.0), focal=5100.0 * scale,
                          W=self.W, H=self.H, cx=(W0 / 2 - x0) * scale, cy=(H0 / 2 + 100 - y0) * scale)
        self.light = Light(toward=LIGHT_DIR, soft=0.06)
        self.lt = Lighting(self.light, self.cam)
        self.cv = Canvas(self.W, self.H, ground=ground, seed=seed)
        self.rng = np.random.default_rng(seed)

    def px(self, v):
        """A size given in full-canvas pixels, at this scene's scale."""
        return v * self.s
