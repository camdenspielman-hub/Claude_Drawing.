"""Assemble the recorded frames of a painting session into a time-lapse GIF and MP4.

    python3 tools/timelapse.py FRAMES_DIR FINAL_IMAGE OUT_BASENAME [--frames 150] [--width 720] [--fps 12]

The painting program saves a frame every so much brushwork (see atelier.canvas.Recorder);
this picks an even selection of them, ends on the finished picture, and holds it a few seconds.
"""
import argparse
import glob
import os
import shutil
import subprocess
import tempfile

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("final_image")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--hold", type=float, default=3.5, help="seconds to rest on the finished picture")
    ap.add_argument("--ease", type=float, default=0.6,
                    help="below 1, hurry through the early broad work and linger on the later fine work")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.frames_dir, "f*.jpg")))
    if not files:
        raise SystemExit("no frames in " + args.frames_dir)
    n = min(args.frames, len(files))
    pick = [files[round((i / max(1, n - 1)) ** args.ease * (len(files) - 1))] for i in range(n)]
    final = Image.open(args.final_image).convert("RGB")
    h = int(round(args.width * final.height / final.width / 2)) * 2
    tmp = tempfile.mkdtemp(prefix="timelapse_")
    try:
        k = 0
        for f in pick:
            Image.open(f).convert("RGB").resize((args.width, h), Image.LANCZOS).save(os.path.join(tmp, "%05d.png" % k))
            k += 1
        last = final.resize((args.width, h), Image.LANCZOS)
        for _ in range(int(args.hold * args.fps)):
            last.save(os.path.join(tmp, "%05d.png" % k))
            k += 1
        src = ["-framerate", str(args.fps), "-i", os.path.join(tmp, "%05d.png")]
        quiet = ["-hide_banner", "-loglevel", "error", "-y"]
        pal = os.path.join(tmp, "palette.png")
        subprocess.run(["ffmpeg", *quiet, *src, "-vf", "palettegen=max_colors=192:stats_mode=diff", pal], check=True)
        subprocess.run(["ffmpeg", *quiet, *src, "-i", pal, "-lavfi", "paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                        "-loop", "0", args.out + ".gif"], check=True)
        subprocess.run(["ffmpeg", *quiet, *src, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                        args.out + ".mp4"], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for ext in (".gif", ".mp4"):
        print(args.out + ext, "%.1f MB" % (os.path.getsize(args.out + ext) / 1e6))


if __name__ == "__main__":
    main()
