#!/usr/bin/env python3
r"""Combine real co-placement snapshots into a synchronized three-panel GIF.

Example (run from the repository root)::

    python3 placer/scripts/make_coplace_gif.py results/uart/<run> \
        --title "UART | synchronized HBT anchors" \
        --output results/uart/<run>/coplace.gif --duration-ms 180

The input must contain matching ``iterNNNN.png`` snapshots under each of
``tier0/co_place/plot``, ``tier1/co_place/plot``, and ``terminal/co_place/plot``.
Snapshot labels use the numeric filename indices, which are not necessarily
completed optimizer iteration counts. ``--last-frame-label`` can identify a
final restored placement. The script does not interpolate positions, fill
missing snapshots, or run placement. Requires Pillow.
"""

import argparse
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont, ImageOps


LAYERS = (("tier0", "Top die"), ("tier1", "Bottom die"),
          ("terminal", "HBT layer"))
FRAME_NAME = re.compile(r"iter(\d+)\.png$")


def collect_frames(root):
    """Reject missing or mismatched iterations instead of skipping them."""
    layers = {}
    for layer, _ in LAYERS:
        directory = root / layer / "co_place" / "plot"
        if not directory.is_dir():
            raise ValueError("snapshot directory does not exist: %s" % directory)
        frames = {}
        for path in directory.glob("iter*.png"):
            match = FRAME_NAME.fullmatch(path.name)
            if match is None:
                raise ValueError("invalid snapshot filename: %s" % path)
            iteration = int(match.group(1))
            if iteration in frames:
                raise ValueError("duplicate iteration %d in %s" %
                                 (iteration, directory))
            frames[iteration] = path
        if not frames:
            raise ValueError("no iterNNNN.png snapshots in %s" % directory)
        layers[layer] = frames

    iterations = set(layers[LAYERS[0][0]])
    for layer, _ in LAYERS[1:]:
        other = set(layers[layer])
        if other != iterations:
            raise ValueError(
                "iteration mismatch for %s versus tier0: missing=%s extra=%s" %
                (layer, sorted(iterations - other), sorted(other - iterations)))
    return layers, sorted(iterations)


def load_font(size):
    for name in ("DejaVuSans.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_frame(layers, iteration, title, panel_width, subtitle=None):
    gutter = 16
    margin = 20
    header_height = 86
    label_height = 36
    width = panel_width * len(LAYERS) + gutter * (len(LAYERS) - 1) + 2 * margin
    height = header_height + label_height + panel_width + margin
    frame = Image.new("RGB", (width, height), "#f5f7fa")
    draw = ImageDraw.Draw(frame)
    title_font = load_font(24)
    label_font = load_font(20)
    title_width = draw.textlength(title, font=title_font)
    # Keep unusually long case names visible without cropping the title.
    if title_width > width - 2 * margin:
        title_font = load_font(max(10, int(24 * (width - 2 * margin) / title_width)))
    draw.text((margin, 15), title, font=title_font, fill="#152238")
    draw.text((margin, 49), subtitle if subtitle is not None else "Snapshot %d" % iteration,
              font=label_font, fill="#526176")

    for index, (layer, label) in enumerate(LAYERS):
        left = margin + index * (panel_width + gutter)
        draw.text((left, header_height), label, font=label_font, fill="#152238")
        top = header_height + label_height
        draw.rectangle((left, top, left + panel_width - 1,
                        top + panel_width - 1), fill="white")
        with Image.open(layers[layer][iteration]) as source:
            # Composite transparency onto white before resizing the plot.
            rgba = source.convert("RGBA")
            opaque = Image.new("RGBA", rgba.size, "white")
            opaque.alpha_composite(rgba)
            plot = ImageOps.contain(opaque.convert("RGB"),
                                    (panel_width, panel_width),
                                    method=Image.Resampling.LANCZOS)
        frame.paste(plot, (left + (panel_width - plot.width) // 2,
                           top + (panel_width - plot.height) // 2))
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path,
                        help="output GIF path (default: RESULT_DIR/coplace.gif)")
    parser.add_argument("--title", default="Co-placement")
    parser.add_argument("--last-frame-label",
                        help="override the final snapshot subtitle, e.g. a restored best placement")
    parser.add_argument("--duration-ms", type=int, default=180,
                        help="duration of each actual snapshot; multiple of 10 (default: 180)")
    parser.add_argument("--panel-width", type=int, default=500,
                        help="maximum plot width and height in pixels (default: 500)")
    args = parser.parse_args()
    if args.duration_ms < 10 or args.duration_ms % 10:
        parser.error("--duration-ms must be a positive multiple of 10")
    if args.panel_width < 100:
        parser.error("--panel-width must be at least 100")
    output = args.output or args.result_dir / "coplace.gif"
    if output.suffix.lower() != ".gif":
        parser.error("--output must end in .gif")

    try:
        layers, iterations = collect_frames(args.result_dir)
        frames = []
        palette = None
        for iteration in iterations:
            subtitle = args.last_frame_label if iteration == iterations[-1] else None
            frame = draw_frame(layers, iteration, args.title, args.panel_width,
                               subtitle=subtitle)
            if palette is None:
                palette = frame.quantize(colors=256, dither=Image.Dither.NONE)
                frames.append(palette)
            else:
                frames.append(frame.quantize(palette=palette,
                                             dither=Image.Dither.NONE))
        output.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(output, save_all=True, append_images=frames[1:],
                       duration=args.duration_ms, loop=0, disposal=2,
                       optimize=False)
        print("Saved %s (%d synchronized frames, snapshot indices %d–%d)" %
              (output, len(iterations), iterations[0], iterations[-1]))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
