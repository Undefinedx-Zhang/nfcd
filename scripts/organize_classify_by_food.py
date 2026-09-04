#!/usr/bin/env python3
"""Sort classify video directories into food-category directories."""

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set

from prepare_foodcd import parse_stem, read_gray_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classify-dir",
        type=Path,
        default=Path("/mnt/sdb/26_zdj/DATA/Annotations/classify"),
    )
    parser.add_argument(
        "--semantic-mask-dir",
        type=Path,
        default=Path("/mnt/sdb/26_zdj/DATA/Annotations/semantic_mask"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def food_by_class_id(class_map_path: Path) -> Dict[int, str]:
    content = json.loads(class_map_path.read_text(encoding="utf-8"))
    return {
        class_id: class_name.rsplit("_", 1)[0]
        for class_name, class_id in content["class_to_id"].items()
        if class_name != "background"
    }


def top_level_video_dirs(classify_dir: Path, food_names: Set[str]) -> Iterable[Path]:
    for path in sorted(classify_dir.iterdir()):
        if path.is_dir() and path.name not in food_names:
            yield path


def video_food(video_id: str, semantic_mask_dir: Path, class_ids: Dict[int, str]) -> str:
    masks = sorted(
        semantic_mask_dir.glob(f"{video_id}_*.png"),
        key=lambda path: parse_stem(path.stem)[1],
    )
    if not masks:
        raise FileNotFoundError(f"No semantic masks found for video {video_id}")

    sampled_masks = [masks[0], masks[-1]]
    food_names: Set[str] = set()
    for mask_path in sampled_masks:
        _, _, pixels = read_gray_png(mask_path)
        food_names.update(class_ids[value] for value in set(pixels) if value in class_ids)
    if len(food_names) != 1:
        raise ValueError(f"Expected one food class for {video_id}, found {sorted(food_names)}")
    return next(iter(food_names))


def repair_symlinks(video_dir: Path, images_dir: Path) -> None:
    for path in video_dir.iterdir():
        if not path.is_symlink():
            continue
        target = images_dir / path.name
        if not target.is_file():
            raise FileNotFoundError(f"Image target not found for {path}: {target}")
        path.unlink()
        path.symlink_to(os.path.relpath(target, path.parent))


def move_video_dir(source: Path, destination: Path) -> None:
    symlink_targets = {
        path.name: path.resolve(strict=True) for path in source.iterdir() if path.is_symlink()
    }
    shutil.move(str(source), str(destination))
    for image_name, target in symlink_targets.items():
        destination_path = destination / image_name
        destination_path.unlink()
        destination_path.symlink_to(os.path.relpath(target, destination_path.parent))


def classify_videos(args: argparse.Namespace) -> List[Dict[str, str]]:
    class_ids = food_by_class_id(args.semantic_mask_dir / "class_map.json")
    food_names = set(class_ids.values())
    direct_videos = list(top_level_video_dirs(args.classify_dir, food_names))
    categorized_videos = [
        video_dir
        for food_name in food_names
        for video_dir in (args.classify_dir / food_name).glob("*")
        if video_dir.is_dir()
    ]
    videos = direct_videos + categorized_videos
    records = []
    for video_dir in videos:
        food_name = video_food(video_dir.name, args.semantic_mask_dir, class_ids)
        if video_dir.parent.name in food_names and video_dir.parent.name != food_name:
            raise ValueError(f"Food directory mismatch for {video_dir}: expected {food_name}")
        records.append({"video_id": video_dir.name, "food": food_name})
    if len({record["video_id"] for record in records}) != len(records):
        raise ValueError("A video directory appears more than once")
    conflicts = [
        args.classify_dir / record["food"] / record["video_id"]
        for record in records
        if args.classify_dir / record["video_id"] in direct_videos
        if (args.classify_dir / record["food"] / record["video_id"]).exists()
    ]
    if conflicts:
        raise FileExistsError(f"Destination directories already exist: {conflicts[:3]}")

    if args.dry_run:
        return records

    for food_name in sorted(food_names):
        (args.classify_dir / food_name).mkdir(exist_ok=True)
    images_dir = args.classify_dir.parent / "images"
    for video_dir in categorized_videos:
        repair_symlinks(video_dir, images_dir)
    for record in records:
        source = args.classify_dir / record["video_id"]
        if source not in direct_videos:
            continue
        destination = args.classify_dir / record["food"] / record["video_id"]
        move_video_dir(source, destination)

    map_path = args.classify_dir / "food_video_map.csv"
    with map_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("video_id", "food"))
        writer.writeheader()
        writer.writerows(records)
    return records


def main() -> None:
    args = parse_args()
    records = classify_videos(args)
    print(json.dumps(Counter(record["food"] for record in records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
