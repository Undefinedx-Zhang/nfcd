#!/usr/bin/env python3
"""Build the FoodCD dataset and per-video inference directories."""

import argparse
import csv
import json
import math
import os
import random
import shutil
import struct
import subprocess
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class Frame:
    video_id: str
    frame_index: int
    stem: str
    image_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/mnt/sdb/26_zdj/DATA/Annotations/images"),
    )
    parser.add_argument(
        "--semantic-mask-dir",
        type=Path,
        default=Path("/mnt/sdb/26_zdj/DATA/Annotations/semantic_mask"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/sdb/26_zdj/DATA/Annotations/FoodCD"),
    )
    parser.add_argument(
        "--classify-dir",
        type=Path,
        default=Path("/mnt/sdb/26_zdj/DATA/Annotations/classify"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--val-video-count", type=int)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument(
        "--classify-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--lists-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--build-classify", action="store_true")
    return parser.parse_args()


def chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def read_png_size(path: Path) -> Tuple[int, int]:
    with path.open("rb") as handle:
        data = handle.read(33)
    if not data.startswith(PNG_SIGNATURE) or data[12:16] != b"IHDR":
        raise ValueError(f"Not a PNG file: {path}")
    return struct.unpack(">II", data[16:24])


def read_gray_png_python(path: Path) -> Tuple[int, int, bytearray]:
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"Not a PNG file: {path}")

    offset = len(PNG_SIGNATURE)
    idat_parts: List[bytes] = []
    width = height = bit_depth = color_type = interlace = None
    while offset < len(data):
        size = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + size]
        offset += size + 12
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
        elif kind == b"IDAT":
            idat_parts.append(payload)
        elif kind == b"IEND":
            break

    if (bit_depth, color_type, interlace) != (8, 0, 0):
        raise ValueError(
            f"{path} must be an 8-bit, grayscale, non-interlaced PNG; got "
            f"bit_depth={bit_depth}, color_type={color_type}, interlace={interlace}"
        )

    raw = zlib.decompress(b"".join(idat_parts))
    stride = width
    expected_size = (stride + 1) * height
    if len(raw) != expected_size:
        raise ValueError(f"Unexpected decoded size for {path}")

    output = bytearray(width * height)
    previous = bytearray(stride)
    position = 0
    for row_index in range(height):
        filter_type = raw[position]
        row = bytearray(raw[position + 1 : position + stride + 1])
        position += stride + 1
        for column in range(stride):
            left = row[column - 1] if column else 0
            up = previous[column]
            upper_left = previous[column - 1] if column else 0
            if filter_type == 1:
                row[column] = (row[column] + left) & 255
            elif filter_type == 2:
                row[column] = (row[column] + up) & 255
            elif filter_type == 3:
                row[column] = (row[column] + (left + up) // 2) & 255
            elif filter_type == 4:
                predictor = left + up - upper_left
                candidates = (left, up, upper_left)
                distances = tuple(abs(predictor - value) for value in candidates)
                row[column] = (row[column] + candidates[distances.index(min(distances))]) & 255
            elif filter_type != 0:
                raise ValueError(f"Unsupported PNG filter {filter_type} in {path}")
        start = row_index * stride
        output[start : start + stride] = row
        previous = row
    return width, height, output


def read_gray_png(path: Path) -> Tuple[int, int, bytes]:
    width, height = read_png_size(path)
    if shutil.which("ffmpeg") is None:
        return read_gray_png_python(path)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    if len(result.stdout) != width * height:
        raise ValueError(f"Unexpected decoded size for {path}")
    return width, height, result.stdout


def write_gray_png(path: Path, width: int, height: int, pixels: bytes) -> None:
    if len(pixels) != width * height:
        raise ValueError(f"Unexpected pixel length while writing {path}")
    raw = bytearray((width + 1) * height)
    for row_index in range(height):
        source_start = row_index * width
        destination_start = row_index * (width + 1)
        raw[destination_start + 1 : destination_start + width + 1] = pixels[
            source_start : source_start + width
        ]
    content = (
        PNG_SIGNATURE
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), level=6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(content)


def read_jpeg_size(path: Path) -> Tuple[int, int]:
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise ValueError(f"Not a JPEG file: {path}")
        while True:
            marker_prefix = handle.read(1)
            while marker_prefix == b"\xff":
                marker_prefix = handle.read(1)
            if not marker_prefix:
                break
            marker = marker_prefix[0]
            if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            segment_length_bytes = handle.read(2)
            if len(segment_length_bytes) != 2:
                break
            segment_length = struct.unpack(">H", segment_length_bytes)[0]
            if marker in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                payload = handle.read(segment_length - 2)
                height, width = struct.unpack(">HH", payload[1:5])
                return width, height
            handle.seek(segment_length - 2, os.SEEK_CUR)
    raise ValueError(f"Unable to read JPEG dimensions: {path}")


def image_size(path: Path) -> Tuple[int, int]:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return read_jpeg_size(path)
    return read_png_size(path)


def parse_stem(stem: str) -> Tuple[str, int]:
    try:
        video_id, frame_index_text = stem.rsplit("_", 1)
        return video_id, int(frame_index_text)
    except ValueError as error:
        raise ValueError(f"Invalid frame filename stem: {stem}") from error


def index_frames(images_dir: Path, semantic_mask_dir: Path) -> Dict[str, List[Frame]]:
    image_paths = {path.stem: path for path in images_dir.glob("*.jpg")}
    mask_paths = {
        path.stem: path
        for path in semantic_mask_dir.glob("*.png")
        if path.name != "class_map.json"
    }
    missing_masks = sorted(set(image_paths) - set(mask_paths))
    missing_images = sorted(set(mask_paths) - set(image_paths))
    if missing_masks or missing_images:
        raise ValueError(
            f"Missing masks={len(missing_masks)} ({missing_masks[:3]}), "
            f"missing images={len(missing_images)} ({missing_images[:3]})"
        )

    groups: Dict[str, List[Frame]] = defaultdict(list)
    for stem, image_path in image_paths.items():
        video_id, frame_index = parse_stem(stem)
        groups[video_id].append(
            Frame(video_id, frame_index, stem, image_path, mask_paths[stem])
        )
    for frames in groups.values():
        frames.sort(key=lambda frame: frame.frame_index)
        indices = [frame.frame_index for frame in frames]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate frame indexes in video {frames[0].video_id}")
    return dict(sorted(groups.items()))


def cooked_ids(class_map_path: Path) -> Set[int]:
    content = json.loads(class_map_path.read_text(encoding="utf-8"))
    class_to_id = content["class_to_id"]
    return {
        class_id
        for class_name, class_id in class_to_id.items()
        if class_name != "background" and not class_name.endswith("_0")
    }


def validate_frames(groups: Dict[str, List[Frame]], cooked_class_ids: Set[int]) -> Dict[str, object]:
    first_frame_issues = []
    total_cooked_pixels = 0
    first_frame_mask_values: Set[int] = set()
    for video_id, frames in groups.items():
        reference = frames[0]
        reference_size = image_size(reference.image_path)
        for frame in (reference, frames[-1]):
            width, height = read_png_size(frame.mask_path)
            if image_size(frame.image_path) != (width, height):
                raise ValueError(f"Image/mask size mismatch for {frame.stem}")
            if image_size(frame.image_path) != reference_size:
                raise ValueError(f"Frame size differs from first frame in {video_id}: {frame.stem}")
        _, _, reference_pixels = read_gray_png(reference.mask_path)
        values = set(reference_pixels)
        first_frame_mask_values.update(values)
        cooked_pixels = sum(value in cooked_class_ids for value in reference_pixels)
        total_cooked_pixels += cooked_pixels
        if cooked_pixels:
            first_frame_issues.append(
                {"video_id": video_id, "frame": reference.image_path.name, "cooked_pixels": cooked_pixels}
            )
    return {
        "videos": len(groups),
        "frames": sum(len(frames) for frames in groups.values()),
        "first_frame_mask_values": sorted(first_frame_mask_values),
        "first_frame_cooked_pixels": total_cooked_pixels,
        "first_frame_cooked": first_frame_issues,
    }


def prepare_path(path: Path, overwrite: bool) -> None:
    if path.exists() or path.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Refusing to replace existing file: {path}")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def materialize(source: Path, destination: Path, mode: str, overwrite: bool) -> None:
    prepare_path(destination, overwrite)
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        try:
            destination.hardlink_to(source)
        except OSError:
            shutil.copy2(source, destination)
    else:
        destination.symlink_to(os.path.relpath(source, destination.parent))


def write_text(path: Path, lines: Iterable[str], overwrite: bool) -> None:
    prepare_path(path, overwrite)
    path.write_text("".join(lines), encoding="utf-8")


def build_splits(
    groups: Dict[str, List[Frame]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    val_video_count: int = None,
) -> Dict[str, List[str]]:
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio > 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to at most one")
    video_ids = list(groups)
    if val_video_count is not None:
        if not 0 < val_video_count < len(video_ids):
            raise ValueError("val_video_count must be between 1 and the total number of videos minus one")
        shuffled_ids = video_ids[:]
        random.Random(seed).shuffle(shuffled_ids)
        validation_ids = set(shuffled_ids[:val_video_count])
        return {
            "train": sorted(set(video_ids) - validation_ids),
            "val": sorted(validation_ids),
            "test": [],
        }
    if math.isclose(train_ratio + val_ratio, 1.0):
        target_frames = round(sum(len(frames) for frames in groups.values()) * val_ratio)
        shuffled_ids = video_ids[:]
        random.Random(seed).shuffle(shuffled_ids)
        reachable = {0: ()}
        for video_id in shuffled_ids:
            frame_count = len(groups[video_id])
            for total, selected in list(reachable.items()):
                candidate_total = total + frame_count
                if candidate_total not in reachable:
                    reachable[candidate_total] = selected + (video_id,)
        validation_total = min(reachable, key=lambda total: (abs(total - target_frames), total))
        validation_ids = set(reachable[validation_total])
        return {
            "train": sorted(set(video_ids) - validation_ids),
            "val": sorted(validation_ids),
            "test": [],
        }
    random.Random(seed).shuffle(video_ids)
    train_count = math.floor(len(video_ids) * train_ratio)
    val_count = math.floor(len(video_ids) * val_ratio)
    return {
        "train": sorted(video_ids[:train_count]),
        "val": sorted(video_ids[train_count : train_count + val_count]),
        "test": sorted(video_ids[train_count + val_count :]),
    }


def write_lists(
    output_dir: Path,
    samples_by_video: Dict[str, List[str]],
    splits: Dict[str, List[str]],
    seed: int,
    overwrite: bool,
) -> None:
    list_dir = output_dir / "list"
    split_manifest = {"seed": seed, "splits": splits}
    prepare_path(list_dir / "split_manifest.json", overwrite)
    (list_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    train_names: List[str] = []
    for split_name, video_ids in splits.items():
        names = sorted(name for video_id in video_ids for name in samples_by_video[video_id])
        write_text(list_dir / f"{split_name}.txt", (f"{name}\n" for name in names), overwrite)
        if split_name == "train":
            train_names = names

    shuffled_train = train_names[:]
    random.Random(seed).shuffle(shuffled_train)
    for percent in (5, 10, 20, 40):
        supervised_count = math.floor(len(shuffled_train) * percent / 100)
        supervised = shuffled_train[:supervised_count]
        unsupervised = shuffled_train[supervised_count:]
        write_text(
            list_dir / f"{percent}_train_supervised.txt",
            (f"{name}\n" for name in supervised),
            overwrite,
        )
        write_text(
            list_dir / f"{percent}_train_unsupervised.txt",
            (f"{name}\n" for name in unsupervised),
            overwrite,
        )


def build_dataset(
    groups: Dict[str, List[Frame]],
    cooked_class_ids: Set[int],
    output_dir: Path,
    copy_mode: str,
    splits: Dict[str, List[str]],
    seed: int,
    overwrite: bool,
    workers: int,
) -> None:
    lookup_table = bytes(255 if value in cooked_class_ids else 0 for value in range(256))
    tasks = [
        (frames[0], frame)
        for frames in groups.values()
        for frame in frames
    ]

    def build_sample(task: Tuple[Frame, Frame]) -> Dict[str, object]:
        reference, frame = task
        image_name = frame.image_path.name
        materialize(reference.image_path, output_dir / "A" / image_name, copy_mode, overwrite)
        materialize(frame.image_path, output_dir / "B" / image_name, copy_mode, overwrite)
        width, height, pixels = read_gray_png(frame.mask_path)
        label_pixels = bytes(pixels).translate(lookup_table)
        label_path = output_dir / "label" / f"{frame.stem}.png"
        prepare_path(label_path, overwrite)
        write_gray_png(label_path, width, height, label_pixels)
        changed_pixels = label_pixels.count(255)
        return {
            "sample_name": image_name,
            "video_id": frame.video_id,
            "frame_index": frame.frame_index,
            "a_source": str(reference.image_path),
            "b_source": str(frame.image_path),
            "semantic_mask_source": str(frame.mask_path),
            "label_path": str(label_path),
            "changed_pixels": changed_pixels,
            "change_ratio": f"{changed_pixels / len(label_pixels):.8f}",
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        samples = list(executor.map(build_sample, tasks))
    samples.sort(key=lambda sample: (str(sample["video_id"]), int(sample["frame_index"])))

    manifest_path = output_dir / "manifest.csv"
    prepare_path(manifest_path, overwrite)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)

    samples_by_video: Dict[str, List[str]] = defaultdict(list)
    for sample in samples:
        samples_by_video[str(sample["video_id"])].append(str(sample["sample_name"]))
    write_lists(output_dir, samples_by_video, splits, seed, overwrite)

    summary = {
        "videos": len(groups),
        "samples": len(samples),
        "cooked_ids": sorted(cooked_class_ids),
        "all_zero_labels": sum(int(sample["changed_pixels"]) == 0 for sample in samples),
        "changed_pixels": sum(int(sample["changed_pixels"]) for sample in samples),
        "splits": {name: len(video_ids) for name, video_ids in splits.items()},
    }
    summary_path = output_dir / "summary.json"
    prepare_path(summary_path, overwrite)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def build_classify(
    groups: Dict[str, List[Frame]], classify_dir: Path, mode: str, overwrite: bool
) -> None:
    manifest_rows = []
    for video_id, frames in groups.items():
        for order, frame in enumerate(frames):
            materialize(frame.image_path, classify_dir / video_id / frame.image_path.name, mode, overwrite)
            manifest_rows.append(
                {
                    "video_id": video_id,
                    "order": order,
                    "frame_index": frame.frame_index,
                    "image_name": frame.image_path.name,
                    "is_reference": int(order == 0),
                }
            )
        video_manifest = classify_dir / video_id / "frames.csv"
        prepare_path(video_manifest, overwrite)
        with video_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[-len(frames)]))
            writer.writeheader()
            writer.writerows(manifest_rows[-len(frames) :])
    root_manifest = classify_dir / "manifest.csv"
    prepare_path(root_manifest, overwrite)
    with root_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)


def main() -> None:
    args = parse_args()
    class_map_path = args.semantic_mask_dir / "class_map.json"
    groups = index_frames(args.images_dir, args.semantic_mask_dir)
    cooked_class_ids = cooked_ids(class_map_path)
    if args.lists_only:
        splits = build_splits(
            groups, args.train_ratio, args.val_ratio, args.seed, args.val_video_count
        )
        manifest_path = args.output_dir / "manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"FoodCD manifest not found: {manifest_path}")
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            samples = list(csv.DictReader(handle))
        samples_by_video: Dict[str, List[str]] = defaultdict(list)
        for sample in samples:
            samples_by_video[sample["video_id"]].append(sample["sample_name"])
        write_lists(args.output_dir, samples_by_video, splits, args.seed, True)
        summary_path = args.output_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["splits"] = {name: len(video_ids) for name, video_ids in splits.items()}
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"splits": splits}, indent=2))
        return
    summary = validate_frames(groups, cooked_class_ids)
    summary["cooked_ids"] = sorted(cooked_class_ids)
    summary["first_frames"] = {
        video_id: frames[0].image_path.name for video_id, frames in groups.items()
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["first_frame_cooked"]:
        raise RuntimeError("Reference frames contain cooked pixels; resolve label semantics before building")
    if args.dry_run:
        return

    splits = build_splits(groups, args.train_ratio, args.val_ratio, args.seed, args.val_video_count)
    build_dataset(
        groups,
        cooked_class_ids,
        args.output_dir,
        args.copy_mode,
        splits,
        args.seed,
        args.overwrite,
        args.workers,
    )
    if args.build_classify:
        build_classify(groups, args.classify_dir, args.classify_mode, args.overwrite)
    print(json.dumps({"output_dir": str(args.output_dir), "splits": splits}, indent=2))


if __name__ == "__main__":
    main()
