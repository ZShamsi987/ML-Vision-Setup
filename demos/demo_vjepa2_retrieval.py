# demos/demo_vjepa2_retrieval.py
# Tiny video retrieval demo (CLIP encoder as a lightweight stand-in).
# Output: JSON with per-video nearest neighbors (+ optional text→video).

import argparse, json, os
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


def list_videos(root: Path) -> List[Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    vids = sorted([p for p in root.iterdir() if p.suffix.lower() in exts])
    if not vids:
        vids = sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
    return vids


def sample_frames_cv2(path: Path, max_frames: int) -> List[Image.Image]:
    """Evenly sample up to `max_frames` RGB PIL frames from a video."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // max_frames) if total and total > max_frames else 1

    frames = []
    idx = taken = 0
    while taken < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            taken += 1
        idx += 1
    cap.release()
    return frames


@torch.no_grad()
def encode_video_frames(frames: List[Image.Image],
                        model: CLIPModel,
                        proc: CLIPProcessor,
                        device: torch.device,
                        batch_size: int = 16) -> torch.Tensor:
    """Return a single L2-normalized embedding for the whole clip."""
    if not frames:
        return torch.zeros(model.config.projection_dim, device=device)

    feats = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        inputs = proc(images=batch, return_tensors="pt").to(device)
        img_feats = model.get_image_features(**inputs)  # (B, D)
        img_feats = torch.nn.functional.normalize(img_feats, dim=-1)
        feats.append(img_feats)
    feats = torch.cat(feats, dim=0)                    # (N, D)
    vid_feat = feats.mean(dim=0, keepdim=False)        # (D,)
    vid_feat = torch.nn.functional.normalize(vid_feat, dim=-1)
    return vid_feat


@torch.no_grad()
def encode_text_queries(queries: List[str],
                        model: CLIPModel,
                        proc: CLIPProcessor,
                        device: torch.device) -> torch.Tensor:
    if not queries:
        return torch.empty(0, model.config.projection_dim, device=device)
    inputs = proc(text=queries, return_tensors="pt", padding=True).to(device)
    txt = model.get_text_features(**inputs)
    return torch.nn.functional.normalize(txt, dim=-1)


def cosine_topk(Q: torch.Tensor, X: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Q: (Nq, D), X: (Nv, D) both normalized -> returns top-k (scores, indices)."""
    sims = Q @ X.T
    scores, idxs = torch.topk(sims, k=min(k, X.shape[0]), dim=1)
    return scores, idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/ucf101_mini")
    ap.add_argument("--model_id", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--text", type=str, default="")  # ';' separated
    ap.add_argument("--out", type=str, default="outputs/vjepa2_retrieval.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[retrieval] device={device}, model={args.model_id}")

    # Force safe weights to avoid torch.load CVE checks on older torch.
    # Transformers will pick *.safetensors if present.
    try:
        model = CLIPModel.from_pretrained(args.model_id, use_safetensors=True).to(device).eval()
    except Exception as e:
        # Last-resort fallback: if the repo truly has no safetensors, you’d need torch>=2.6.
        raise SystemExit(
            f"Failed to load safetensors for {args.model_id} ({e}). "
            "Either choose a model with safetensors or upgrade torch to >=2.6."
        )
    proc = CLIPProcessor.from_pretrained(args.model_id)

    # 1) Load & embed videos
    videos = list_videos(data_dir)
    if not videos:
        raise SystemExit(f"No videos found under {data_dir}.")
    print(f"[retrieval] found {len(videos)} videos")

    vid_embs = []
    rel_paths = []
    for vp in tqdm(videos, desc="embed videos"):
        frames = sample_frames_cv2(vp, max_frames=args.frames)
        emb = encode_video_frames(frames, model, proc, device)
        vid_embs.append(emb)
        try:
            rel_paths.append(str(vp.relative_to(data_dir)))
        except Exception:
            rel_paths.append(str(vp))

    V = torch.stack(vid_embs, dim=0)  # (Nv, D)

    # 2) Video→Video nearest neighbors (remove self)
    scores_vv, idxs_vv = cosine_topk(V, V, k=args.topk + 1)
    vv_results: List[Dict] = []
    for qi in range(len(videos)):
        neigh = []
        for j in range(1, min(args.topk + 1, idxs_vv.shape[1])):  # skip self at j=0
            idx = int(idxs_vv[qi, j])
            neigh.append({"video": rel_paths[idx], "score": float(scores_vv[qi, j])})
        vv_results.append({"video": rel_paths[qi], "neighbors": neigh})

    # 3) Optional Text→Video
    tv_results: List[Dict] = []
    text_queries = [t.strip() for t in args.text.split(";") if t.strip()] if args.text else []
    if text_queries:
        T = encode_text_queries(text_queries, model, proc, device)
        scores_tv, idxs_tv = cosine_topk(T, V, k=args.topk)
        for qi, q in enumerate(text_queries):
            hits = []
            for j in range(idxs_tv.shape[1]):
                idx = int(idxs_tv[qi, j])
                hits.append({"video": rel_paths[idx], "score": float(scores_tv[qi, j])})
            tv_results.append({"query": q, "topk": hits})

    # 4) Save JSON
    payload = {
        "encoder": args.model_id,
        "device": str(device),
        "frames_per_video": args.frames,
        "num_videos": len(videos),
        "video_neighbors": vv_results,
        "text2video": tv_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[retrieval] wrote {out_path}")


if __name__ == "__main__":
    main()