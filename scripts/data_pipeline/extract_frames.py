"""
Extract frames from dashcam footage at a fixed rate (default 1 FPS).
Usage: python scripts/data_pipeline/extract_frames.py --input data/raw/dashcam.mp4 --fps 1
"""

import argparse
import os
import cv2
from pathlib import Path


def extract_frames(video_path: str, output_dir: str, fps: float = 1.0, resize: tuple = (640, 360)):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(source_fps / fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Source FPS: {source_fps:.1f} | Extracting every {frame_interval} frames ({fps} FPS)")
    print(f"Total source frames: {total_frames} | Estimated output: {total_frames // frame_interval}")

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if resize:
                frame = cv2.resize(frame, resize)
            out_path = output_dir / f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Saved {saved} frames to {output_dir}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", default="data/frames", help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()

    extract_frames(args.input, args.output, args.fps, (args.width, args.height))
