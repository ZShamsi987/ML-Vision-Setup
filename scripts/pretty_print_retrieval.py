#!/usr/bin/env python
import json, argparse, os
from pathlib import Path

def bar(title):
    print("\n" + title)
    print("-" * len(title))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=str, help="Path to retrieval JSON")
    ap.add_argument("--width", type=int, default=80, help="Terminal wrap width")
    args = ap.parse_args()

    p = Path(args.json_path)
    if not p.exists():
        raise SystemExit(f"File not found: {p}")

    data = json.loads(p.read_text())

    enc = data.get("encoder")
    dev = data.get("device")
    nvid = data.get("num_videos")
    frames = data.get("frames_per_video")
    print(f"[summary] encoder={enc} | device={dev} | videos={nvid} | frames/vid={frames}")

    # Video→Video neighbors
    vv = data.get("video_neighbors", [])
    if vv:
        bar("Video → Video (nearest neighbors)")
        for item in vv:
            vid = item["video"]
            neigh = item.get("neighbors", [])
            line = f"{vid:>20}  ->  "
            line += " | ".join([f"{n['video']} ({n['score']:.3f})" for n in neigh])
            print(line)

    # Text→Video (if present)
    tv = data.get("text2video", [])
    if tv:
        bar("Text → Video (top-k)")
        for q in tv:
            qstr = q["query"]
            hits = q.get("topk", [])
            print(f"'{qstr}':")
            for h in hits:
                print(f"  - {h['video']:>20}  ({h['score']:.3f})")

if __name__ == "__main__":
    main()