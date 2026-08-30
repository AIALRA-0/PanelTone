from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rectangle(box, fill="white", outline="#111111", width=7)


def face(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    hair: str = "short",
) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="white", outline="#111111", width=6)
    if hair == "long":
        draw.arc((x - radius - 10, y - radius - 18, x + radius + 10, y + radius + 35), 170, 370, fill="#111111", width=24)
        draw.arc((x - radius, y - radius - 10, x + radius, y + radius), 190, 350, fill="#111111", width=18)
    else:
        draw.arc((x - radius, y - radius - 12, x + radius, y + radius), 195, 345, fill="#111111", width=22)
    draw.ellipse((x - radius // 3 - 5, y - 6, x - radius // 3 + 5, y + 5), fill="#111111")
    draw.ellipse((x + radius // 3 - 5, y - 6, x + radius // 3 + 5, y + 5), fill="#111111")
    draw.arc((x - 25, y + 12, x + 25, y + 42), 10, 170, fill="#111111", width=4)


def halftone(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], spacing: int = 14) -> None:
    left, top, right, bottom = box
    for y in range(top, bottom, spacing):
        offset = spacing // 2 if (y // spacing) % 2 else 0
        for x in range(left + offset, right, spacing):
            draw.ellipse((x, y, x + 2, y + 2), fill="#777777")


def create_page(index: int, target: Path) -> None:
    image = Image.new("RGB", (900, 1200), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 18, 882, 1182), outline="#111111", width=8)
    draw.text((48, 40), f"PANELTONE  /  SYNTHETIC PAGE {index + 1:02d}", fill="#111111", font=font(24, True))

    panel(draw, (45, 90, 855, 490))
    halftone(draw, (52, 97, 848, 483), 16)
    for building in range(7):
        x = 55 + building * 115
        height = 120 + ((building * 47 + index * 31) % 190)
        draw.rectangle((x, 480 - height, x + 92, 480), fill="white", outline="#111111", width=5)
        for wy in range(480 - height + 18, 465, 32):
            for wx in range(x + 14, x + 80, 26):
                draw.rectangle((wx, wy, wx + 10, wy + 14), fill="#111111")
    draw.rounded_rectangle((80, 118, 355, 230), radius=44, fill="white", outline="#111111", width=6)
    draw.polygon(((310, 220), (355, 270), (335, 214)), fill="white", outline="#111111")
    draw.text((115, 145), "LOCAL. PRIVATE. FAST.", fill="#111111", font=font(20, True))
    draw.text((115, 178), "THE PAGE STAYS YOURS.", fill="#111111", font=font(16))

    panel(draw, (45, 520, 430, 890))
    face(draw, (235, 670), 95, "long" if index % 2 else "short")
    draw.line((160, 760, 112, 855), fill="#111111", width=8)
    draw.line((310, 760, 360, 855), fill="#111111", width=8)
    draw.line((160, 760, 310, 760), fill="#111111", width=8)
    for line in range(5):
        draw.line((80, 560 + line * 22, 125, 545 + line * 18), fill="#111111", width=3)

    panel(draw, (470, 520, 855, 890))
    face(draw, (660, 670), 95, "short" if index % 2 else "long")
    draw.line((585, 760, 540, 855), fill="#111111", width=8)
    draw.line((735, 760, 790, 855), fill="#111111", width=8)
    draw.line((585, 760, 735, 760), fill="#111111", width=8)
    for ray in range(8):
        angle = ray * math.pi / 4
        start = (660 + int(math.cos(angle) * 125), 670 + int(math.sin(angle) * 125))
        end = (660 + int(math.cos(angle) * 165), 670 + int(math.sin(angle) * 165))
        draw.line((start, end), fill="#111111", width=3)

    panel(draw, (45, 920, 855, 1145))
    draw.rounded_rectangle((90, 950, 410, 1080), radius=55, fill="white", outline="#111111", width=6)
    draw.polygon(((375, 1065), (435, 1110), (405, 1048)), fill="white", outline="#111111")
    draw.text((135, 980), "INK AND TEXT", fill="#111111", font=font(28, True))
    draw.text((145, 1020), "remain unchanged", fill="#111111", font=font(22))
    draw.text((520, 975), "PAGE READY", fill="#111111", font=font(30, True))
    draw.text((520, 1020), "as soon as it finishes", fill="#111111", font=font(21))
    for x in range(520, 795, 28):
        draw.line((x, 1085, x + 20, 1120), fill="#111111", width=4)

    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("assets/demo/source"))
    parser.add_argument("--pages", type=int, default=6)
    args = parser.parse_args()
    for index in range(args.pages):
        create_page(index, args.output / f"page_{index + 1:03d}.png")


if __name__ == "__main__":
    main()
