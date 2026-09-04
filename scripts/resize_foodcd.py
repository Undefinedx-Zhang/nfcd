#!/usr/bin/env python3
"""Create a resized FoodCD training copy for faster online data loading."""

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/mnt/sdb/26_zdj/DATA/Annotations/FoodCD"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/sdb/26_zdj/DATA/Annotations/FoodCD_512"),
    )
    parser.add_argument("--long-side", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def scale_filter(long_side: int, flags: str) -> str:
    return (
        f"scale={long_side}:{long_side}:force_original_aspect_ratio=decrease:"
        f"force_divisible_by=2:flags={flags}"
    )


def run_ffmpeg(source: Path, destination: Path, video_filter: str, pixel_format: str) -> None:
    temporary_path = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-vf",
        video_filter,
        "-pix_fmt",
        pixel_format,
    ]
    if destination.suffix.lower() == ".jpg":
        command.extend(["-q:v", "2"])
    command.append(str(temporary_path))
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        os.replace(temporary_path, destination)
    except subprocess.CalledProcessError as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {source}: {error.stderr.strip()}") from error


def image_key(path: Path) -> Tuple[int, int]:
    metadata = path.stat()
    return metadata.st_dev, metadata.st_ino


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def output_paths(source_dir: Path) -> Iterable[Tuple[Path, Path]]:
    for directory_name in ("A", "B", "label"):
        source_subdir = source_dir / directory_name
        for source_path in sorted(source_subdir.iterdir()):
            if source_path.is_file():
                yield source_path, Path(directory_name) / source_path.name


def resize_dataset(args: argparse.Namespace) -> Dict[str, int]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to create the resized dataset")
    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {args.source_dir}")
    if args.output_dir.exists():
        raise FileExistsError(f"Output path already exists: {args.output_dir}")
    if args.long_side < 2:
        raise ValueError("--long-side must be at least 2")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    files = list(output_paths(args.source_dir))
    image_sources: Dict[Tuple[int, int], Tuple[Path, List[Path]]] = {}
    label_sources: List[Tuple[Path, Path]] = []
    for source_path, relative_path in files:
        if relative_path.parts[0] == "label":
            label_sources.append((source_path, relative_path))
            continue
        key = image_key(source_path)
        if key not in image_sources:
            image_sources[key] = (source_path, [])
        image_sources[key][1].append(relative_path)

    summary = {
        "image_outputs": sum(len(paths) for _, paths in image_sources.values()),
        "unique_images": len(image_sources),
        "labels": len(label_sources),
        "long_side": args.long_side,
    }
    if args.dry_run:
        return summary

    for directory_name in ("A", "B", "label"):
        (args.output_dir / directory_name).mkdir(parents=True, exist_ok=True)

    image_filter = scale_filter(args.long_side, "lanczos")
    label_filter = scale_filter(args.long_side, "neighbor")

    def resize_image(item: Tuple[Path, List[Path]]) -> None:
        source_path, relative_paths = item
        canonical_path = args.output_dir / relative_paths[0]
        run_ffmpeg(source_path, canonical_path, image_filter, "yuvj420p")
        for relative_path in relative_paths[1:]:
            link_or_copy(canonical_path, args.output_dir / relative_path)

    def resize_label(item: Tuple[Path, Path]) -> None:
        source_path, relative_path = item
        run_ffmpeg(source_path, args.output_dir / relative_path, label_filter, "gray")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(resize_image, image_sources.values()))
        list(executor.map(resize_label, label_sources))

    for filename in ("manifest.csv", "summary.json"):
        shutil.copy2(args.source_dir / filename, args.output_dir / filename)
    shutil.copytree(args.source_dir / "list", args.output_dir / "list")
    (args.output_dir / "preprocess.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(resize_dataset(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
