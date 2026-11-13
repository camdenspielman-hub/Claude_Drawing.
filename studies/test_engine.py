"""First marks: check that the engine lays, blends, glazes and dry-brushes paint."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from atelier.canvas import Canvas, Paint, Brush

c = Canvas(1200, 800, ground="#7d6648")
t0 = time.time()
ochre, white, umber = Paint.opaque("#c89a3c"), Paint.opaque("#f4efe2"), Paint.opaque("#3a2616")
blue, red = Paint.opaque("#2f4f7d"), Paint.opaque("#a82a1c")

# 1. opaque strokes, soft round brush at several hardnesses
for i, h in enumerate([0.1, 0.5, 0.9]):
    x = np.linspace(60, 380, 40); y = 60 + i * 50 + 10 * np.sin(x / 40)
    p = np.clip(np.minimum(np.arange(40) / 6, (39 - np.arange(40)) / 6), 0, 1)
    c.stroke(np.stack([x, y], 1), white, Brush(radius=14, hardness=h, thick=1.5), p)

# 2. bristle strokes: loaded, and running dry
for i, (load, tooth) in enumerate([(0, 0.0), (9, 0.0), (5, 0.8)]):
    x = np.linspace(440, 1140, 60); y = 60 + i * 55 + 8 * np.sin(x / 60)
    c.stroke(np.stack([x, y], 1), ochre, Brush(kind="bristle", radius=18, aspect=0.35, angle=np.pi / 2,
             nbristle=70, bristle_r=1.3, jitter=0.25, thick=1.2, load=load, tooth=tooth), 0.9)

# 3. wet-in-wet: two colour fields, then drag a clean brush across the join
for j in range(26):
    y = 250 + j * 6
    c.stroke([[60, y], [300, y]], blue, Brush(radius=7, hardness=0.5, thick=1.5), 1.0)
    c.stroke([[300, y], [540, y]], white, Brush(radius=7, hardness=0.5, thick=1.5), 1.0)
for j in range(8):
    y = 262 + j * 18
    x = np.linspace(120, 480, 30)
    c.stroke(np.stack([x, y + 5 * np.sin(x / 15)], 1), white,
             Brush(radius=10, hardness=0.3, thick=0.0, pickup=0.6, load=0.001), 1.0)
# bristle blending lower half
for j in range(5):
    y = 340 + j * 14
    c.stroke([[150, y], [450, y + 6]], white, Brush(kind="bristle", radius=12, nbristle=50, bristle_r=1.2,
             thick=0.0, pickup=0.7, load=0.001), 1.0)

# 4. glazes over a dried gradient of white strokes
for j in range(30):
    y = 250 + j * 5.5
    c.stroke([[620, y], [1140, y]], Paint.opaque((0.25 + j / 40, 0.25 + j / 40, 0.25 + j / 40)),
             Brush(radius=6, hardness=0.5, thick=1.5), 1.0)
c.dry_now()
for i, (col, x0) in enumerate([("#b02010", 640), ("#d8a020", 770), ("#204890", 900), ("#3a2a18", 1030)]):
    for k in range(3):
        c.stroke([[x0 + 15 + k * 30, 250], [x0 + 15 + k * 30, 415]], Paint.glaze(col, 1.2),
                 Brush(radius=18, hardness=0.4, thick=0.4 * (k + 1), pickup=0.0), 1.0)

# 5. scumble: thin white, bristles, over the dark ground, tooth breaking it up
dark = Paint.opaque("#201810")
for j in range(22):
    y = 470 + j * 7
    c.stroke([[60, y], [560, y]], dark, Brush(radius=8, hardness=0.5, thick=1.5), 1.0)
c.dry_now()
for j in range(6):
    y = 490 + j * 22
    c.stroke([[80, y], [540, y + 4]], white, Brush(kind="bristle", radius=16, nbristle=60, bristle_r=1.2,
             thick=0.25 + 0.15 * j, tooth=0.9, load=6, jitter=0.1), 0.35 + 0.1 * j)

# 6. thick highlight dabs (impasto) and hairlines
for i in range(8):
    c.stroke([[640 + i * 60, 520], [660 + i * 60, 512]], white, Brush(radius=2 + i, hardness=0.8, thick=4), [0.3, 1.0])
for i in range(6):
    x = np.linspace(640, 1140, 50); y = 580 + i * 22 + 6 * np.sin(x / 30 + i)
    c.stroke(np.stack([x, y], 1), white, Brush(radius=0.5 + 0.35 * i, hardness=0.9, thick=1.5, press_radius=0.6),
             np.clip(np.sin(np.linspace(0, np.pi, 50)) * 1.3, 0, 1))
print("strokes", c.nstrokes, "time %.2fs" % (time.time() - t0))
out = os.path.join(os.path.dirname(__file__), "out"); os.makedirs(out, exist_ok=True)
t0 = time.time(); c.save(os.path.join(out, "test_engine.png"), relief=0.35, tooth_amt=0.15); print("render %.2fs" % (time.time() - t0))
