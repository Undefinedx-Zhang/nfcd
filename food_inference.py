#!/usr/bin/env python3
"""Infer FoodCD changes for one image pair or every frame in a video directory."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from tqdm import tqdm
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
    parser.add_argument(
        "--gt-mask-dir",
        type=Path,
        default=Path("/mnt/sdb/26_zdj/DATA/Annotations/semantic_mask"),
        help="Semantic-mask directory for per-frame GT metrics; use --no-gt to disable.",
    )
    parser.add_argument(
        "--gt-class-map",
        type=Path,
        default=None,
        help="Class-map JSON; defaults to <gt-mask-dir>/class_map.json.",
    )
    parser.add_argument("--pixel-prob-threshold", type=float, default=0.5)
    parser.add_argument("--change-iou-threshold", type=float, default=0.5)
    parser.add_argument("--no-gt", action="store_true", help="Disable GT change metrics.")
    parser.add_argument(
        "--inference-long-side",
        type=int,
        default=None,
        help=(
            "Resize the longer input-image side before inference; "
            "defaults to train_supervised.base_size."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.video_dir is None and (args.image_a is None or args.image_b is None):
        parser.error("Specify both --image-a and --image-b, or specify --video-dir")
    if args.video_dir is not None and (args.image_a is not None or args.image_b is not None):
        parser.error("--video-dir cannot be combined with --image-a or --image-b")
    if not 0 <= args.pixel_prob_threshold <= 1:
        parser.error("--pixel-prob-threshold must be between 0 and 1")
    if not 0 <= args.change_iou_threshold <= 1:
        parser.error("--change-iou-threshold must be between 0 and 1")
    if args.inference_long_side is not None and args.inference_long_side < 1:
        parser.error("--inference-long-side must be positive")
    if args.no_gt:
        if args.gt_class_map is not None:
            parser.error("--gt-class-map cannot be combined with --no-gt")
        args.gt_mask_dir = None
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
    return model_classes[backbone](num_classes=2, config=config, testing=True, pretrained=None)


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


def load_cooked_class_ids(class_map_path: Path) -> Set[int]:
    class_map = json.loads(class_map_path.read_text(encoding="utf-8"))
    return {
        class_id
        for class_name, class_id in class_map["class_to_id"].items()
        if class_name != "background" and not class_name.endswith("_0")
    }


def change_metrics(
    prediction_mask: np.ndarray,
    gt_mask_path: Path,
    cooked_class_ids: Set[int],
    change_iou_threshold: float,
) -> Dict[str, object]:
    gt_mask = np.asarray(Image.open(gt_mask_path))
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[:, :, 0]
    gt_change = np.isin(gt_mask, list(cooked_class_ids))
    prediction_change = prediction_mask > 0
    if prediction_change.shape != gt_change.shape:
        raise ValueError(
            f"Prediction and GT sizes differ: {prediction_change.shape} != {gt_change.shape}"
        )

    changed_pixels = int(np.count_nonzero(gt_change))
    predicted_pixels = int(np.count_nonzero(prediction_change))
    intersection = int(np.count_nonzero(prediction_change & gt_change))
    union = int(np.count_nonzero(prediction_change | gt_change))
    change_iou = intersection / union if union else None
    precision = intersection / predicted_pixels if predicted_pixels else None
    recall = intersection / changed_pixels if changed_pixels else None
    f1 = None
    if precision is not None and recall is not None and precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    change_iou_pass = change_iou is not None and change_iou >= change_iou_threshold
    gt_change_ratio = changed_pixels / gt_change.size
    return {
        "gt_changed_pixels": changed_pixels,
        "gt_change_ratio": gt_change_ratio,
        "gt_has_mature_mask": bool(changed_pixels),
        "change_iou": change_iou,
        "change_iou_pass": change_iou_pass,
        "precision_change": precision,
        "recall_change": recall,
        "f1_change": f1,
    }


def load_pair(
    image_a_path: Path,
    image_b_path: Path,
    inference_long_side: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    image_a = Image.open(image_a_path).convert("RGB")
    image_b = Image.open(image_b_path).convert("RGB")
    if image_a.size != image_b.size:
        raise ValueError(f"Image sizes differ: {image_a_path}={image_a.size}, {image_b_path}={image_b.size}")
    width, height = image_a.size
    if inference_long_side is not None and max(width, height) != inference_long_side:
        scale = inference_long_side / max(width, height)
        resized_size = (round(width * scale), round(height * scale))
        image_a = image_a.resize(resized_size, Image.BICUBIC)
        image_b = image_b.resize(resized_size, Image.BICUBIC)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
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
    input_height, input_width = image_a.shape[-2:]
    padded_height = ((input_height + 7) // 8) * 8
    padded_width = ((input_width + 7) // 8) * 8
    pad_height = padded_height - input_height
    pad_width = padded_width - input_width
    if pad_height or pad_width:
        image_a = functional.pad(image_a, (0, pad_width, 0, pad_height), mode="reflect")
        image_b = functional.pad(image_b, (0, pad_width, 0, pad_height), mode="reflect")
    with torch.no_grad():
        logits = model(A_l=image_a, B_l=image_b)
        if logits.shape[-2:] != (padded_height, padded_width):
            logits = functional.interpolate(
                logits, size=(padded_height, padded_width), mode="bilinear", align_corners=True
            )
        probabilities = torch.softmax(logits, dim=1)[0, 1, :input_height, :input_width]
        if probabilities.shape != original_size:
            probabilities = functional.interpolate(
                probabilities.unsqueeze(0).unsqueeze(0),
                size=original_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
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
) -> Dict[str, object]:
    result_dir.mkdir(parents=True, exist_ok=True)
    changed_pixels = int(np.count_nonzero(prediction_mask))
    total_pixels = int(prediction_mask.size)
    change_ratio = changed_pixels / total_pixels
    result = {
        "image_a": str(image_a_path),
        "image_b": str(image_b_path),
        "pixel_probability_threshold": pixel_threshold,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "change_ratio": change_ratio,
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
    inference_long_side: int,
    gt_mask_path: Optional[Path] = None,
    cooked_class_ids: Optional[Set[int]] = None,
    change_iou_threshold: float = 0.5,
) -> Dict[str, object]:
    image_a, image_b, original_size = load_pair(
        image_a_path,
        image_b_path,
        inference_long_side,
    )
    probability_map, prediction_mask = predict(
        model, image_a, image_b, original_size, device, pixel_threshold
    )
    result = save_result(
        result_dir,
        image_a_path,
        image_b_path,
        probability_map,
        prediction_mask,
        pixel_threshold,
    )
    if gt_mask_path is not None and cooked_class_ids is not None:
        result.update(
            change_metrics(
                prediction_mask,
                gt_mask_path,
                cooked_class_ids,
                change_iou_threshold,
            )
        )
    return result


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    inference_long_side = args.inference_long_side
    if inference_long_side is None:
        inference_long_side = config["train_supervised"].get("base_size")
    if not isinstance(inference_long_side, int):
        raise ValueError(
            "--inference-long-side is required when train_supervised.base_size is not an integer"
        )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(config, args.model, device)
    cooked_class_ids = None
    if args.gt_mask_dir is not None:
        class_map_path = args.gt_class_map or args.gt_mask_dir / "class_map.json"
        cooked_class_ids = load_cooked_class_ids(class_map_path)
    if args.video_dir is None:
        gt_mask_path = None
        if args.gt_mask_dir is not None:
            gt_mask_path = args.gt_mask_dir / f"{args.image_b.stem}.png"
        result = infer_pair(
            model,
            args.image_a,
            args.image_b,
            args.output_dir,
            device,
            args.pixel_prob_threshold,
            inference_long_side,
            gt_mask_path,
            cooked_class_ids,
            args.change_iou_threshold,
        )
        print(json.dumps(result, indent=2))
        return

    frame_paths = sorted(args.video_dir.glob("*.jpg"), key=frame_sort_key)
    if not frame_paths:
        raise FileNotFoundError(f"No JPG frames in {args.video_dir}")
    reference = frame_paths[0]
    results: List[Dict[str, object]] = []
    progress_bar = tqdm(
        enumerate(frame_paths, start=1),
        total=len(frame_paths),
        desc=f"Infer {args.video_dir.name}",
        unit="frame",
    )
    for frame_order, frame_path in progress_bar:
        gt_mask_path = None
        if args.gt_mask_dir is not None:
            gt_mask_path = args.gt_mask_dir / f"{frame_path.stem}.png"
        result = infer_pair(
            model,
            reference,
            frame_path,
            args.output_dir / frame_path.stem,
            device,
            args.pixel_prob_threshold,
            inference_long_side,
            gt_mask_path,
            cooked_class_ids,
            args.change_iou_threshold,
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
        "pixel_probability_threshold",
        "image_a",
        "image_b",
    ]
    if args.gt_mask_dir is not None:
        fieldnames.extend(
            [
                "gt_changed_pixels",
                "gt_change_ratio",
                "gt_has_mature_mask",
                "change_iou",
                "change_iou_pass",
                "precision_change",
                "recall_change",
                "f1_change",
            ]
        )
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    excel_path = args.output_dir / "maturity_results.xlsx"
    write_excel(excel_path, fieldnames, results)
    summary = {
        "reference_frame": reference.name,
        "frames": len(results),
        "pixel_probability_threshold": args.pixel_prob_threshold,
    }
    if args.gt_mask_dir is not None:
        first_gt_mature = next((result for result in results if result["gt_has_mature_mask"]), None)
        first_iou_mature = next((result for result in results if result["change_iou_pass"]), None)
        valid_iou_results = [result for result in results if result["change_iou"] is not None]
        mature_iou_results = [
            result
            for result in valid_iou_results
            if result["gt_has_mature_mask"]
        ]
        summary.update(
            {
                "gt_mask_dir": str(args.gt_mask_dir),
                "change_iou_threshold": args.change_iou_threshold,
                "change_iou_evaluated_frames": len(valid_iou_results),
                "mean_change_iou": (
                    sum(result["change_iou"] for result in valid_iou_results)
                    / len(valid_iou_results)
                    if valid_iou_results
                    else None
                ),
                "gt_mature_mask_frames": len(mature_iou_results),
                "mean_gt_mature_change_iou": (
                    sum(result["change_iou"] for result in mature_iou_results)
                    / len(mature_iou_results)
                    if mature_iou_results
                    else None
                ),
                "gt_mature_mask_start_pair": (
                    first_gt_mature["pair"] if first_gt_mature else None
                ),
                "gt_mature_mask_start_frame": (
                    first_gt_mature["frame_name"] if first_gt_mature else None
                ),
                "iou_maturity_start_pair": (
                    first_iou_mature["pair"] if first_iou_mature else None
                ),
                "iou_maturity_start_frame": (
                    first_iou_mature["frame_name"] if first_iou_mature else None
                ),
            }
        )
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
