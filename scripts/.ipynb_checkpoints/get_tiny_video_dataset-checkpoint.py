# scripts/get_tiny_video_dataset.py
# Download a tiny set of UCF101 videos without decoding (no torchcodec needed).
# Uses huggingface_hub to fetch only the video files and copies the first N.

import argparse
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=str, default="data/ucf101_mini")
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.dest)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pull only video files from the dataset repo (avoid any decoding)
    repo_id = "sayakpaul/ucf101-subset"
    print(f"Fetching a tiny subset from '{repo_id}' …")
    snap_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["*.avi", "*.mp4"],   # only fetch actual video files
    )

    # Collect videos and copy the first N to dest with canonical names
    vids = sorted(list(Path(snap_dir).rglob("*.avi")) + list(Path(snap_dir).rglob("*.mp4")))
    if not vids:
        raise SystemExit("No videos found in the snapshot. (Unexpected for this repo)")

    take = min(args.limit, len(vids))
    saved = []
    for i, src in enumerate(vids[:take]):
        dst = out_dir / f"{i:05d}{src.suffix.lower()}"
        shutil.copy(src, dst)
        saved.append(str(dst))

    print(f"Saved {len(saved)} videos to {out_dir}:")
    for p in saved:
        print(p)

if __name__ == "__main__":
    main()