from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def create_demo_book(destination: Path, pages: int = 3) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for index in range(pages):
        image = Image.new("RGB", (640, 900), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((36, 36, 604, 864), outline="black", width=8)
        draw.rectangle((64, 64, 576, 480), outline="black", width=5)
        draw.ellipse((150, 110, 490, 450), outline="black", width=7)
        draw.ellipse((235, 230, 260, 255), fill="black")
        draw.ellipse((380, 230, 405, 255), fill="black")
        draw.arc((245, 270, 400, 355), 15, 165, fill="black", width=6)
        draw.rounded_rectangle((115, 540, 525, 720), radius=55, outline="black", width=7)
        draw.text((218, 615), f"DEMO PAGE {index + 1}", fill="black")
        draw.line((80, 790, 560, 790), fill="black", width=5)
        image.save(destination / f"page_{index + 1:03d}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--pages", type=int, default=3)
    args = parser.parse_args()
    create_demo_book(args.destination, args.pages)


if __name__ == "__main__":
    main()
