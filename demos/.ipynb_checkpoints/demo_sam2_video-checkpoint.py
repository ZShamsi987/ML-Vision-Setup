# demos/demo_sam2_video.py
# SAM 2 video demo: overlay masks on frames, pad to even dims, then encode with ffmpeg CLI.
import argparse, os, shutil, cv2, numpy as np, subprocess
from pathlib import Path
import torch
from tqdm import tqdm
from sam2.sam2_video_predictor import SAM2VideoPredictor


def extract_frames(video_path, frames_dir, max_frames=90, stride=1):
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i = saved = 0
    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            cv2.imwrite(os.path.join(frames_dir, f"{saved}.jpg"), frame)
            saved += 1
        i += 1
    cap.release()
    return int(round(fps)) if fps > 0 else 25, saved


def auto_point(first_frame_path):
    img = cv2.imread(first_frame_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    sat = hsv[..., 1].astype(np.float32) / 255.0
    cx, cy = w / 2, h / 2
    wx = np.exp(-((xs - cx) ** 2) / (2 * (0.25 * w) ** 2))
    wy = np.exp(-((ys - cy) ** 2) / (2 * (0.25 * h) ** 2))
    heat = sat * wx * wy
    y, x = np.unravel_index(np.argmax(heat), heat.shape)
    return np.array([[x, y]], dtype=np.float32), np.array([1], dtype=np.int32)


def _overlay_frame(frame_bgr, mask_bool):
    """Overlay green mask + cyan edges onto a BGR frame."""
    if mask_bool is None:
        return frame_bgr
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    overlay = (0.35 * np.array([0, 255, 0], dtype=np.uint8) + 0.65 * frame).astype(np.uint8)
    frame = np.where(mask_bool[..., None], overlay, frame)
    edges = cv2.Canny((mask_bool.astype(np.uint8) * 255), 50, 150)
    frame[edges > 0] = [0, 255, 255]
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def _pad_even_min2(img_bgr):
    """Pad image so width/height are even and at least 2 px (libx264/yuv420p requirement)."""
    h, w = img_bgr.shape[:2]
    pad_bottom = (2 - (h % 2)) % 2
    pad_right  = (2 - (w % 2)) % 2
    if h < 2:
        pad_bottom += (2 - h)
    if w < 2:
        pad_right += (2 - w)
    if pad_bottom or pad_right:
        img_bgr = cv2.copyMakeBorder(img_bgr, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)
    return img_bgr


def write_video_ffmpeg(png_dir: Path, out_path: str, fps: int):
    """Encode {png_dir}/%05d.png -> MP4 using ffmpeg CLI, forcing even dims (pad)."""
    png_glob = str(png_dir / "%05d.png")
    out_dir = Path(os.path.dirname(out_path) or ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-start_number", "0",
        "-framerate", str(fps),
        "-i", png_glob,
        # pad to even dims to satisfy yuv420p
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"[SAM2] Saved {out_path}")
    except FileNotFoundError:
        raise SystemExit("ffmpeg not found. Install it with: sudo apt install -y ffmpeg")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"ffmpeg failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default="data/ucf101_mini/00000.avi")
    ap.add_argument("--hf_id", type=str, default="facebook/sam2.1-hiera-small")
    ap.add_argument("--max_frames", type=int, default=90)
    ap.add_argument("--workdir", type=str, default="work/sam2_frames")
    ap.add_argument("--out", type=str, default="outputs/sam2_demo.mp4")
    ap.add_argument("--keep_pngs", action="store_true", help="Keep overlaid PNG frames (for debugging).")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        raise SystemExit(f"Video not found: {args.video}. Run the tiny dataset script first.")

    # 1) Extract frames
    if os.path.exists(args.workdir):
        shutil.rmtree(args.workdir)
    fps, n = extract_frames(args.video, args.workdir, max_frames=args.max_frames)
    assert n > 0, "No frames extracted."

    # 2) Load model
    try:
        predictor = SAM2VideoPredictor.from_pretrained(args.hf_id)
    except Exception:
        predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-small")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # 3) Propagate masks
    with torch.inference_mode(), torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=amp_dtype):
        state = predictor.init_state(args.workdir)
        pt, lbl = auto_point(os.path.join(args.workdir, "0.jpg"))
        _, _, _ = predictor.add_new_points_or_box(state, frame_idx=0, points=pt, labels=lbl, obj_id=1)

        segs = {}
        for frame_idx, ids, mask_logits in tqdm(predictor.propagate_in_video(state), total=n, desc="propagate in video"):
            # merge all object masks into a single boolean mask for overlay
            acc = None
            for i, oid in enumerate(ids):
                m = (mask_logits[i] > 0.0).cpu().numpy()
                acc = m if acc is None else np.maximum(acc, m)
            segs[frame_idx] = acc

    # 4) Render overlaid PNG frames (padded to even dims)
    png_dir = Path(args.workdir) / "overlaid"
    png_dir.mkdir(parents=True, exist_ok=True)
    frame_files = sorted([f for f in os.listdir(args.workdir) if f.lower().endswith(".jpg")],
                         key=lambda p: int(os.path.splitext(p)[0]))
    for name in frame_files:
        idx = int(os.path.splitext(name)[0])
        bgr = cv2.imread(os.path.join(args.workdir, name))
        over = _overlay_frame(bgr, segs.get(idx, None))
        over = _pad_even_min2(over)
        cv2.imwrite(str(png_dir / f"{idx:05d}.png"), over)

    # 5) Encode with ffmpeg
    write_video_ffmpeg(png_dir, args.out, fps)

    # 6) Cleanup
    if not args.keep_pngs:
        shutil.rmtree(png_dir, ignore_errors=True)


if __name__ == "__main__":
    main()