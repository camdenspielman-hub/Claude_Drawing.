# The Cut Thread

A still life painted stroke by stroke in code: a cat's skull, an hourglass, a ball of yarn with
its needles, a heap of mixed coins, half an orange, and a dagger driven into the table.

![The Cut Thread, second version](out/still_life.jpg)

A time-lapse of this version being painted is in [`out/timelapse.gif`](out/timelapse.gif)
(sharper: [`out/timelapse.mp4`](out/timelapse.mp4)).

This is the second version. The first is kept for comparison:
[`out/first_pass.jpg`](out/first_pass.jpg), with its own
[time-lapse](out/first_pass_timelapse.gif).

## What is in the picture

The three Fates of Greek myth spin the thread of a life, measure it, and cut it. Here the yarn
is spun, the hourglass that measured has been knocked on its side, and the blade has come down
through the strand where it crossed the table. The arrangement is what a cat's play leaves
behind, long after the cat: coins scattered, the glass toppled, the yarn dragged across
everything. The strand is still caught behind the fang of the cat's own skull.

## The rules it was made under

- No image model. Nothing is generated, traced or copied from a photograph or an earlier picture.
- Every mark on the canvas is a brush stroke laid through a simulated oil-paint engine.
- The objects are constructed the way a draughtsman or a modeller builds them: in perspective,
  from measured sections or from lumps of "clay" pressed together, checked against photographs
  of the real things. Nothing is drawn into the picture ahead of the brush.

What that does and does not mean, plainly. Before a stroke is laid, the program asks its model
of the scene a painter's questions about that one spot: which way does the surface face, does
it see the window, how shut in is it by its neighbours, what colour do they throw onto it, what
would I see mirrored in it? That is a hand-written light logic, queried per stroke. It is not
a renderer producing an image that is then copied, and no image of the scene exists until the
strokes make one; but it is also not a painter's eye. It is nearer to painting from a clay
model under a known lamp than to painting from life.

Reference photographs (cat skulls, sand-glasses, daggers, hammered coins, wool, cut oranges,
oak, and still lifes by Claesz and Heda) were studied, from Wikimedia Commons, Openverse and
museum collections. None is included here.

## What changed in the second version, and why

The first version read as a painting of an old 3D render. The causes, and what was done:

| Cause | Remedy |
|---|---|
| Front-lit: every surface we see faced the window alike | The window moved round to the side, so forms turn from light into shadow across what we see; darker room, wider range of tone |
| Forms were ruler-and-compass solids stuck together | The skull and the ball of yarn are now *sculpted solids* (`atelier/solid.py`): bone runs into bone, hollows are real, and each shadows itself |
| Each object painted on its own, to the same finish | The whole picture is brought along together in the old order of work (`atelier/studio.py`): drawing, dead colouring, colour, modelling, detail, glaze, finish |
| Miniature brushes everywhere, so everything was airbrushed | Broad brushes first; small ones only where the form needs them |
| Objects ignored each other | Soft cast shadows, contact shadows, colour thrown between neighbours, and reflections of the room in metal and glass (`atelier/shading.py`) |
| Every edge equally sharp | Edges are judged one by one against what is on the canvas: kept where light meets dark near the skull, lost where like meets like |
| Nothing ever looked at the canvas | A correcting pass compares the canvas with what should be there and goes back in where they differ |

Three rounds of adversarial review by separate critic agents (a painting instructor, a
render-tell hunter and a natural-history curator, each told to tear the picture apart) drove
most of the specific corrections: the skull's proportions and teeth, the hourglass turned so its
glass could catch the window, the dagger's attached shadow, the strand cut and frayed at the
blade and tucked behind the fang, the collars of pale wall round objects, and more.

## What still falls short

SHORTFALLS

## How it works

| Part | What it does |
|---|---|
| `engine/paint.cpp` | The paint. A canvas with a dry layer and a wet film; brushes that lay paint down and pick it up; Kubelka-Munk optics, so thin paint is translucent and thick paint covers. |
| `engine/solid.cpp` | Sculpted solids: lumps and rods joined and carved smoothly, and a wound ball of strand. Answers "where does this line of sight touch you, which way do you face there, do you see the window, how shut in are you?" |
| `atelier/` | The toolkit: perspective and constructed shadows (`geom.py`), the light logic (`shading.py`), constructed forms (`forms.py`), sculpted ones (`solid.py`), outlines (`stencil.py`), ways of laying strokes over a form (`brushwork.py`), and the stages of a picture (`studio.py`). |
| `still_life/` | The picture: the set-up (`scene.py`), wall and table (`setting.py`), one file per object, and `paint.py`, which runs the stages. |
| `studies/` | Studies: engine tests, the first block-ins, and `skull_turn.py`, which lays the sculpted skull in broadly from the side, above, front and back to check its anatomy. |
| `tools/timelapse.py` | Turns the frames recorded during painting into the GIF and MP4. |

## Painting it again

```bash
sh engine/build.sh
```

```bash
python3 -m still_life.paint --out out/still_life.png --frames frames
```

```bash
python3 tools/timelapse.py frames out/still_life.png out/timelapse
```

It needs a C++ compiler with OpenMP, Python 3 with NumPy and Pillow, and `ffmpeg` for the
time-lapse. The full canvas is 3000 x 2000 pixels: RUNSTATS.

The time-lapse shows one complete run of the finished program, from bare ground to the last
stroke. It does not show the many earlier attempts the program was revised through.

To work on one object, paint it alone in a window of the canvas:

```bash
python3 -m still_life.paint --only skull --scale 0.5 --out studies/out/skull.png
```
