# demos/demo_grounding_dino.py
# Grounding DINO via Hugging Face Transformers (no repo build needed).
import argparse, os
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
import requests
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def _font():
    try:
        return ImageFont.truetype("arial.ttf", 18)
    except Exception:
        return ImageFont.load_default()

def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """Cross-version text measurement (Pillow 10/11 compatible)."""
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    if hasattr(font, "getbbox"):
        l, t, r, b = font.getbbox(text)
        return r - l, b - t
    if hasattr(draw, "textsize"):  # Pillow <11
        return draw.textsize(text, font=font)
    # very conservative fallback
    return int(len(text) * 0.6 * font.size), font.size

def draw_boxes(image: Image.Image, boxes, labels, scores, save_path: str):
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font()

    drew_any = False
    for (xmin, ymin, xmax, ymax), label, score in zip(boxes, labels, scores):
        drew_any = True
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0, 255, 0, 255), width=3)
        text = f"{label} {score:.2f}"
        tw, th = _text_size(draw, text, font)
        bg = [(xmin, max(0, ymin - th - 4)), (xmin + tw + 6, ymin)]
        draw.rectangle(bg, fill=(0, 255, 0, 200))
        draw.text((bg[0][0] + 3, bg[0][1] + 2), text, fill=(0, 0, 0, 255), font=font)

    out = Image.alpha_composite(img, overlay).convert("RGB")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(save_path)
    msg = "[GroundingDINO] Saved {}{}".format(
        save_path, "" if drew_any else " (no detections above thresholds)"
    )
    print(msg)

def load_image_from_arg(arg_path: str) -> Image.Image:
    if arg_path.startswith("http"):
        return Image.open(requests.get(arg_path, stream=True).raw).convert("RGB")
    if arg_path and os.path.exists(arg_path):
        return Image.open(arg_path).convert("RGB")
    # default COCO sample
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    return Image.open(requests.get(url, stream=True).raw).convert("RGB")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default="",
                    help="Path or URL. Defaults to a COCO sample image.")
    ap.add_argument("--text", type=str, default="a person. a dog. a ball.",
                    help="Dot-separated phrases, e.g., 'a person. a bicycle.'")
    ap.add_argument("--out", type=str, default="outputs/grounding_dino.jpg")
    ap.add_argument("--model", type=str, default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--box_threshold", type=float, default=0.35)
    ap.add_argument("--text_threshold", type=float, default=0.25)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # prompts
    phrases = [t.strip() for t in args.text.split(".") if t.strip()]
    text_for_model = [phrases]  # list-of-list keeps label ids stable

    # image
    image = load_image_from_arg(args.image)

    # model
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()

    inputs = processor(images=image, text=text_for_model, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=[image.size[::-1]],  # (h, w)
    )[0]

    boxes = results["boxes"].tolist()
    scores = results["scores"].tolist()
    if "text_labels" in results and results["text_labels"] is not None:
        labels = [str(x) for x in results["text_labels"]]
    else:
        idx2phrase = {i: (phrases[i] if i < len(phrases) else f"id_{i}") for i in set(results["labels"])}
        labels = [idx2phrase[i] for i in results["labels"]]

    draw_boxes(image, boxes, labels, scores, args.out)

if __name__ == "__main__":
    main()