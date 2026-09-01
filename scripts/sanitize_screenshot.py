from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    args.target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(args.source) as image:
        image.convert("RGB").save(args.target, format="PNG", optimize=True)


if __name__ == "__main__":
    main()
