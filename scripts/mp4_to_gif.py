#!/usr/bin/env python3
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print("Command failed:\n", " ".join(cmd), "\n\nOutput:\n", p.stdout, file=sys.stderr)
        raise SystemExit(p.returncode)


def file_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def make_gif_ffmpeg(
    in_mp4: Path,
    out_gif: Path,
    start: float | None,
    duration: float | None,
    fps: int,
    width: int,
    colors: int,
    dither: str = "bayer",
    bayer_scale: int = 5,
) -> None:
    # scale keeps aspect ratio: width:-1
    vf_palette = f"fps={fps},scale={width}:-1:flags=lanczos,palettegen=max_colors={colors}"
    vf_gif = (
        f"fps={fps},scale={width}:-1:flags=lanczos[x];"
        f"[x][1:v]paletteuse=dither={dither}:bayer_scale={bayer_scale}"
    )

    with tempfile.TemporaryDirectory() as td:
        palette = Path(td) / "palette.png"

        base_in = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        if start is not None:
            base_in += ["-ss", str(start)]
        base_in += ["-i", str(in_mp4)]
        if duration is not None:
            base_in += ["-t", str(duration)]

        # 1) palette
        run(base_in + ["-vf", vf_palette, str(palette)])

        # 2) gif using palette
        base_in2 = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        if start is not None:
            base_in2 += ["-ss", str(start)]
        base_in2 += ["-i", str(in_mp4)]
        if duration is not None:
            base_in2 += ["-t", str(duration)]

        run(
            base_in2
            + ["-i", str(palette), "-lavfi", vf_gif, "-loop", "0", str(out_gif)]
        )


def main():
    ap = argparse.ArgumentParser(description="Convert MP4 to size-bounded GIF (palette-based).")
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--max-mb", type=float, default=15.0, help="Target max GIF size in MB (default 15).")
    ap.add_argument("--start", type=float, default=None, help="Start time in seconds (optional).")
    ap.add_argument("--duration", type=float, default=None, help="Duration in seconds (optional).")
    ap.add_argument("--try-fast", action="store_true", help="Prefer smaller output quickly (more aggressive settings).")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    # Candidate settings from higher quality -> smaller.
    # Tweak these if you want: fps, width(px), colors
    candidates = [
        (15, 960, 128),
        (15, 840, 112),
        (12, 840, 96),
        (12, 720, 96),
        (10, 720, 80),
        (10, 640, 64),
        (8,  640, 64),
        (8,  560, 48),
    ]
    if args.try_fast:
        candidates = [(12, 720, 80), (10, 640, 64), (8, 560, 48), (6, 480, 48)]

    # Ensure output dir exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    best = None
    for fps, width, colors in candidates:
        tmp_out = args.output.with_suffix(".tmp.gif")
        try:
            make_gif_ffmpeg(
                args.input, tmp_out,
                start=args.start, duration=args.duration,
                fps=fps, width=width, colors=colors,
                dither="bayer", bayer_scale=5
            )
        except SystemExit:
            if tmp_out.exists():
                tmp_out.unlink(missing_ok=True)
            raise

        size = file_mb(tmp_out)
        print(f"Try fps={fps}, width={width}, colors={colors} -> {size:.2f} MB")

        if best is None or size < best[0]:
            best = (size, fps, width, colors, tmp_out)

        if size <= args.max_mb:
            tmp_out.replace(args.output)
            print(f"✅ Saved: {args.output} ({size:.2f} MB)")
            return

    # If we didn't hit the target, still save the smallest we got
    if best is not None:
        size, fps, width, colors, tmp_out = best
        tmp_out.replace(args.output)
        print(f"⚠️ Could not reach <= {args.max_mb} MB. Saved smallest: {args.output} ({size:.2f} MB)")
        print(f"   Best settings: fps={fps}, width={width}, colors={colors}")
    else:
        raise SystemExit("Failed to generate GIF.")


if __name__ == "__main__":
    main()
