#!/usr/bin/env python3
"""Infer FoodCD changes for one image pair or every frame in a video directory."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision import transforms

import models


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image-a", type=Path)
    parser.add_argument("--image-b", type=Path)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--pixel-prob-threshold", type=float, default=0.5)
    parser.add_argument("--change-ratio-threshold", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.video_dir is None and (args.image_a is None or args.image_b is None):
        parser.error("Specify both --image-a and --image-b, or specify --video-dir")
    if args.video_dir is not None and (args.image_a is not None or args.image_b is not None):
        parser.error("--video-dir cannot be combined with --image-a or --image-b")
    if not 0 <= args.pixel_prob_threshold <= 1:
        parser.error("--pixel-prob-threshold must be between 0 and 1")
    if not 0 <= args.change_ratio_threshold <= 1:
        parser.error("--change-ratio-threshold must be between 0 and 1")
    return args


def create_model(config: Dict[str, object]) -> torch.nn.Module:
    backbone = config["model"]["backbone"]
    model_classes = {
        "ResNet50": models.NF_ResNet50_CD,
        "ResNet101": models.NF_ResNet101_CD,
        "HRNet": models.NF_HRNet_CD,
        "NF": models.NF_ResNet50_CD,
    }
    if backbone not in model_classes:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model_classes[backbone](num_classes=2, config=config, testing=True, pretrained=False)


def load_model(config: Dict[str, object], checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def load_pair(image_a_path: Path, image_b_path: Path) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    image_a = Image.open(image_a_path).convert("RGB")
    image_b = Image.open(image_b_path).convert("RGB")
    if image_a.size != image_b.size:
        raise ValueError(f"Image sizes differ: {image_a_path}={image_a.size}, {image_b_path}={image_b.size}")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    width, height = image_a.size
    return transform(image_a).unsqueeze(0), transform(image_b).unsqueeze(0), (height, width)


def predict(
    model: torch.nn.Module,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    original_size: Tuple[int, int],
    device: torch.device,
    pixel_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    image_a = image_a.to(device)
    image_b = image_b.to(device)
    height, width = original_size
    padded_height = ((height + 7) // 8) * 8
    padded_width = ((width + 7) // 8) * 8
    pad_height = padded_height - height
    pad_width = padded_width - width
    if pad_height or pad_width:
        image_a = functional.pad(image_a, (0, pad_width, 0, pad_height), mode="reflect")
        image_b = functional.pad(image_b, (0, pad_width, 0, pad_height), mode="reflect")
    with torch.no_grad():
        logits = model(A_l=image_a, B_l=image_b)
        if logits.shape[-2:] != (padded_height, padded_width):
            logits = functional.interpolate(
                logits, size=(padded_height, padded_width), mode="bilinear", align_corners=True
            )
        probabilities = torch.softmax(logits, dim=1)[0, 1, :height, :width]
    probability_map = probabilities.cpu().numpy()
    prediction_mask = (probability_map >= pixel_threshold).astype(np.uint8) * 255
    return probability_map, prediction_mask


def save_result(
    result_dir: Path,
    image_a_path: Path,
    image_b_path: Path,
    probability_map: np.ndarray,
    prediction_mask: np.ndarray,
    pixel_threshold: float,
    ratio_threshold: float,
) -> Dict[str, object]:
    result_dir.mkdir(parents=True, exist_ok=True)
    changed_pixels = int(np.count_nonzero(prediction_mask))
    total_pixels = int(prediction_mask.size)
    change_ratio = changed_pixels / total_pixels
    result = {
        "image_a": str(image_a_path),
        "image_b": str(image_b_path),
        "pixel_probability_threshold": pixel_threshold,
        "change_ratio_threshold": ratio_threshold,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "change_ratio": change_ratio,
        "has_change": change_ratio >= ratio_threshold,
    }
    Image.fromarray(prediction_mask, mode="L").save(result_dir / "prediction_mask.png")
    Image.fromarray(np.round(probability_map * 255).astype(np.uint8), mode="L").save(
        result_dir / "change_probability.png"
    )
    image_b = Image.open(image_b_path).convert("RGB")
    overlay_color = Image.new("RGB", image_b.size, (255, 0, 0))
    alpha = Image.fromarray((prediction_mask > 0).astype(np.uint8) * 128, mode="L")
    Image.composite(overlay_color, image_b, alpha).save(result_dir / "overlay.png")
    (result_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def frame_sort_key(path: Path) -> Tuple[str, int]:
    video_id, frame_index = path.stem.rsplit("_", 1)
    return video_id, int(frame_index)


def excel_column_name(column_index: int) -> str:
    name = ""
    while column_index:
        column_index, remainder = divmod(column_index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def worksheet_cell(cell_ref: str, value: object) -> str:
    if isinstance(value, bool):
        return f'<c r="{cell_ref}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)):
        return f'<c r="{cell_ref}"><v>{value}</v></c>'
    return f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def write_excel(path: Path, headers: List[str], rows: List[Dict[str, object]]) -> None:
    worksheet_rows = []
    for row_index, values in enumerate([dict(zip(headers, headers)), *rows], start=1):
        cells = [
            worksheet_cell(f"{excel_column_name(column_index)}{row_index}", values[header])
            for column_index, header in enumerate(headers, start=1)
        ]
        worksheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(worksheet_rows)}</sheetData></worksheet>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Maturity Results" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    workbook_relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_relationships)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)


def infer_pair(
    model: torch.nn.Module,
    image_a_path: Path,
    image_b_path: Path,
    result_dir: Path,
    device: torch.device,
    pixel_threshold: float,
    ratio_threshold: float,
) -> Dict[str, object]:
    image_a, image_b, original_size = load_pair(image_a_path, image_b_path)
    probability_map, prediction_mask = predict(
        model, image_a, image_b, original_size, device, pixel_threshold
    )
    return save_result(
        result_dir,
        image_a_path,
        image_b_path,
        probability_map,
        prediction_mask,
        pixel_threshold,
        ratio_threshold,
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(config, args.model, device)
    if args.video_dir is None:
        result = infer_pair(
            model,
            args.image_a,
            args.image_b,
            args.output_dir,
            device,
            args.pixel_prob_threshold,
            args.change_ratio_threshold,
        )
        print(json.dumps(result, indent=2))
        return

    frame_paths = sorted(args.video_dir.glob("*.jpg"), key=frame_sort_key)
    if not frame_paths:
        raise FileNotFoundError(f"No JPG frames in {args.video_dir}")
    reference = frame_paths[0]
    results: List[Dict[str, object]] = []
    for frame_order, frame_path in enumerate(frame_paths, start=1):
        result = infer_pair(
            model,
            reference,
            frame_path,
            args.output_dir / frame_path.stem,
            device,
            args.pixel_prob_threshold,
            args.change_ratio_threshold,
        )
        result["frame_name"] = frame_path.name
        result["pair"] = f"1-{frame_order}"
        result["frame_order"] = frame_order
        results.append(result)
    fieldnames = [
        "pair",
        "frame_order",
        "frame_name",
        "changed_pixels",
        "total_pixels",
        "change_ratio",
        "has_change",
        "pixel_probability_threshold",
        "change_ratio_threshold",
        "image_a",
        "image_b",
    ]
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    excel_path = args.output_dir / "maturity_results.xlsx"
    write_excel(excel_path, fieldnames, results)
    first_mature = next((result for result in results if result["has_change"]), None)
    summary = {
        "reference_frame": reference.name,
        "frames": len(results),
        "first_mature_pair": first_mature["pair"] if first_mature else None,
        "first_mature_frame": first_mature["frame_name"] if first_mature else None,
        "pixel_probability_threshold": args.pixel_prob_threshold,
        "change_ratio_threshold": args.change_ratio_threshold,
    }
    summary_path = args.output_dir / "maturity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                **summary,
                "results_csv": str(args.output_dir / "results.csv"),
                "results_excel": str(excel_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
