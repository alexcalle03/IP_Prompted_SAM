import os
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

if os.environ.get("DISPLAY"):
    matplotlib.use("TkAgg", force=True)
else:
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

MODEL_NAME = "facebook/sam-vit-huge"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def list_image_files(folder: str | Path) -> list[Path]:
    folder_path = Path(folder)
    if not folder_path.exists():
        return []
    images = sorted(
        (path for path in folder_path.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda p: str(p).lower(),
    )
    print(f"Found {len(images)} image(s) in folder: {folder_path}")
    return images


def load_image_array(image_path: str | Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def build_visual_canvas(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    points: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    canvas = np.asarray(image, dtype=np.uint8).copy()
    if canvas.ndim == 2:
        canvas = np.repeat(canvas[:, :, None], 3, axis=2)
    if mask is not None:
        overlay = canvas.copy()
        mask_color = np.array([255, 144, 0], dtype=np.uint8)
        overlay[mask] = np.clip(overlay[mask] * 0.5 + mask_color * 0.5, 0, 255).astype(np.uint8)
        canvas = overlay
    if points:
        height, width = canvas.shape[:2]
        for x, y in points:
            x = int(np.clip(x, 0, width - 1))
            y = int(np.clip(y, 0, height - 1))
            y0 = max(0, y - 6)
            y1 = min(height, y + 7)
            x0 = max(0, x - 6)
            x1 = min(width, x + 7)
            canvas[y0:y1, x0:x1] = np.array([255, 255, 255], dtype=np.uint8)
    return canvas


def save_mask_image(mask: np.ndarray, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask_array = np.asarray(mask, dtype=np.uint8)
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]
    mask_image = Image.fromarray((mask_array * 255).astype(np.uint8), mode="L")
    mask_image.save(output)
    return output


def save_mask_overlay(
    image: np.ndarray | Image.Image,
    mask: np.ndarray,
    output_path: str | Path,
    points: list[tuple[int, int]] | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        canvas = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    else:
        canvas = np.asarray(image, dtype=np.uint8).copy()
        if canvas.ndim == 2:
            canvas = np.repeat(canvas[:, :, None], 3, axis=2)
    Image.fromarray(build_visual_canvas(canvas, mask, points), mode="RGB").save(output)
    return output


def resolve_key_action(key: str | None) -> str:
    actions = {"a": "approve", "enter": "approve", "r": "redo", "backspace": "redo", "n": "next", "right": "next", "space": "next", "q": "quit", "escape": "quit"}
    return actions.get((key or "").strip().lower(), "")


_SAM_MODEL: SamModel | None = None
_SAM_PROCESSOR: SamProcessor | None = None


def load_model_and_processor() -> tuple[SamModel, SamProcessor]:
    global _SAM_MODEL, _SAM_PROCESSOR
    if _SAM_MODEL is not None and _SAM_PROCESSOR is not None:
        return _SAM_MODEL, _SAM_PROCESSOR

    model = SamModel.from_pretrained(MODEL_NAME)
    SamModel.cuda(model)
    processor = SamProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    _SAM_MODEL = model
    _SAM_PROCESSOR = processor
    return model, processor


def segment_with_points(
    model: SamModel,
    processor: SamProcessor,
    image: Image.Image,
    points: list[tuple[int, int]],
) -> np.ndarray:
    if not points:
        raise ValueError("At least 1 point is required for segmentation.")

    image_tensor = processor(images=image, return_tensors="pt")["pixel_values"].to(DEVICE)
    image_embeddings = model.get_image_embeddings(image_tensor)
    inputs = processor(
        images=image,
        input_points=[points],
        input_labels=[[1] * len(points)],
        return_tensors="pt",
    ).to(DEVICE)
    inputs.pop("pixel_values", None)
    inputs["image_embeddings"] = image_embeddings

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"],
    )[0]

    if masks.ndim == 4:
        masks = masks[0]
    if masks.ndim == 3 and masks.shape[0] == 1:
        masks = masks[0]
    if masks.ndim == 3 and masks.shape[0] > 1:
        masks = masks[int(outputs.iou_scores.reshape(-1).argmax().item())]

    return masks.detach().cpu().numpy().astype(bool)


def segment_image_with_sam(image: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    model, processor = load_model_and_processor()
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    return segment_with_points(model, processor, pil_image, points)


def _interactive_segmentation(image: np.ndarray, image_path: Path) -> tuple[list[tuple[int, int]], np.ndarray, str]:
    points: list[tuple[int, int]] = []
    mask = np.zeros(image.shape[:2], dtype=bool)
    fig, ax = plt.subplots(figsize=(8, 8))
    status = ""
    preview_ready = False

    def redraw() -> None:
        ax.clear()
        ax.imshow(build_visual_canvas(image, mask, points), interpolation="nearest")
        if preview_ready:
            title = "Mask preview | Enter=save | r=redo | n=next | q=quit"
        else:
            title = "Click points | Enter=run SAM | r=redo | n=next | q=quit"
        ax.set_title(title)
        ax.axis("off")
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if len(points) >= 5:
            points[-1] = (x, y)
        else:
            points.append((x, y))
        mask[:] = False
        preview_ready = False
        redraw()

    def on_key(event) -> None:
        nonlocal status, preview_ready
        action = resolve_key_action(event.key)
        if not action:
            return
        if action == "redo":
            points.clear(); mask[:] = False; preview_ready = False; redraw(); status = "redo"; return
        if action == "approve":
            if len(points) < 3:
                return
            if not preview_ready:
                mask[:] = segment_image_with_sam(image, points)
                preview_ready = True
                redraw()
                return
            status = "approve"
            plt.close(fig)
            return
        if action == "next":
            status = "next"
            plt.close(fig)
            return
        if action == "quit":
            status = "quit"
            plt.close(fig)
            return

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show(block=True)

    if status == "redo":
        return _interactive_segmentation(image, image_path)
    if status == "approve":
        return points, mask, "approve"
    if status == "next":
        return points, mask, "next"
    return points, mask, "quit"


def save_overlay(image: Image.Image, mask: np.ndarray, output_path: str | Path, points: list[tuple[int, int]]) -> Path:
    return save_mask_overlay(image, mask, output_path, points)


