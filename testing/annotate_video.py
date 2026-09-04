#!/usr/bin/env python3
"""
Run a video through the REAL detection pipeline and write it back annotated.

This deliberately reproduces what raspi_scripts/camera/camera_sender.py does to
a live RTSP frame, so the result reflects what the deployed system would see
rather than what the model can do under ideal conditions:

  * rotate to the true orientation (phone video carries a rotation flag that
    OpenCV does not apply, so frames decode sideways)
  * downscale to the camera's working width
  * JPEG-encode at quality 50 — the same lossy compression the Pi applies
  * POST to the live AI service, one frame at a time, with a single camera_id
    so the server-side IoU tracker sees a coherent sequence and can estimate
    speed exactly as it would in production

Deliberately isolated from production data: it uses its own camera_id, and
speed_limit=0 so the service generates no speeding incidents. Frames posted to
/api/detect are never persisted — only camera_sender writes to the database —
so this cannot pollute detections, incidents, or drive the roadside signs.
"""

import argparse
import time

import cv2
import requests

JPEG_QUALITY = 50  # matches camera_sender.JPEG_QUALITY


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--out", default="annotated.mp4")
    ap.add_argument("--ai-url", default="http://100.120.27.110:8000")
    ap.add_argument("--confidence", type=float, default=0.25)
    ap.add_argument("--width", type=int, default=640,
                    help="downscale width, matching the camera substream")
    ap.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="ccw")
    ap.add_argument("--every", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--camera-id", default="VIDEO-TEST")
    args = ap.parse_args()

    rot = {
        "cw": cv2.ROTATE_90_CLOCKWISE,
        "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "180": cv2.ROTATE_180,
    }.get(args.rotate)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    session = requests.Session()
    stats = {"frames": 0, "with_det": 0, "dets": 0}
    by_class: dict = {}
    conf_hist: list = []
    t0 = time.time()

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % args.every:
            continue

        if rot is not None:
            frame = cv2.rotate(frame, rot)

        h, w = frame.shape[:2]
        scale = args.width / w
        frame = cv2.resize(frame, (args.width, int(h * scale)))

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            continue

        dets = []
        try:
            r = session.post(
                f"{args.ai_url}/api/detect",
                files={"image": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                data={
                    "camera_id": args.camera_id,
                    "confidence_threshold": str(args.confidence),
                    "pixels_per_meter": "8.0",
                    "speed_limit": "0",  # no incidents generated
                },
                timeout=30,
            )
            dets = r.json().get("detections", [])
        except Exception as exc:
            print(f"  frame {idx}: request failed: {exc}", flush=True)

        stats["frames"] += 1
        stats["dets"] += len(dets)
        if dets:
            stats["with_det"] += 1

        for d in dets:
            b = d["bbox"]
            x, y = int(b["x"]), int(b["y"])
            bw, bh = int(b["width"]), int(b["height"])
            conf = d["confidence"]
            by_class[d["class"]] = by_class.get(d["class"], 0) + 1
            conf_hist.append(conf)

            # Green above the production threshold, amber below it — so a
            # near-miss is visibly distinct from something the live system
            # would actually have acted on.
            colour = (0, 220, 0) if conf >= 0.5 else (0, 190, 255)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), colour, 2)

            label = f"{d['class']} {conf:.2f}"
            if d.get("speed"):
                label += f" {d['speed']:.0f}km/h"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), colour, -1)
            cv2.putText(frame, label, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        hud = f"frame {idx}/{total}  det:{len(dets)}  conf>={args.confidence}"
        cv2.putText(frame, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if writer is None:
            fh, fw = frame.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     src_fps / args.every, (fw, fh))
        writer.write(frame)

        if stats["frames"] % 50 == 0:
            el = time.time() - t0
            print(f"  {stats['frames']} frames, {stats['dets']} detections, "
                  f"{el:.0f}s elapsed", flush=True)

    cap.release()
    if writer:
        writer.release()

    el = time.time() - t0
    print("\n=== SUMMARY ===")
    print(f"  processed        : {stats['frames']} frames in {el:.0f}s")
    print(f"  frames with a box: {stats['with_det']} "
          f"({100 * stats['with_det'] / max(stats['frames'], 1):.0f}%)")
    print(f"  total detections : {stats['dets']}")
    if by_class:
        print("  by class         : " + ", ".join(f"{k}={v}" for k, v in
                                                  sorted(by_class.items(), key=lambda kv: -kv[1])))
    if conf_hist:
        conf_hist.sort()
        n = len(conf_hist)
        above = sum(1 for c in conf_hist if c >= 0.5)
        print(f"  confidence       : min={conf_hist[0]:.2f} "
              f"median={conf_hist[n // 2]:.2f} max={conf_hist[-1]:.2f}")
        print(f"  above prod 0.50  : {above}/{n} ({100 * above / n:.0f}%)")
    print(f"  output           : {args.out}")


if __name__ == "__main__":
    main()
