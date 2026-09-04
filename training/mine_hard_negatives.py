#!/usr/bin/env python3
"""
Mine hard-negative training images from real Busay footage.

WHY THIS EXISTS
---------------
The deployed model boxes a guardrail post as `truck 0.51` and logs a motionless
"vehicle" at ~(555,345) around eight times a second. No public dataset can fix
that: PH Vehicles, Thammasat and UA-DETRAC contain no picture of OUR guardrail.
The literature is consistent that the durable fix is hard-negative mining — run
inference, collect the false positives, fold them back in as background images —
and that raising the inference threshold only masks the cause.

A background image is simply an image with an EMPTY label file. It teaches the
model what this road with nothing on it looks like, which is exactly the
knowledge it currently lacks.

HOW A CANDIDATE IS FOUND
------------------------
On a camera bolted to a pole, a detection whose box does not move for seconds is
not a moving vehicle. This tracks detections by IoU across frames and flags
clusters whose centroid drifts less than MAX_DRIFT_PX over at least
MIN_TRACK_SECS. Those are fixed scenery: posts, signs, reflectors, markings.

THE TRAP, AND WHY THIS SCRIPT WILL NOT AUTO-ACCEPT
--------------------------------------------------
A genuinely stopped vehicle is ALSO motionless, and `stopped_vehicle` is an
incident this system is required to detect. Labelling those as background would
train the model to ignore precisely the hazard it exists to warn about. So the
static test alone can never decide. `scan` only proposes; a human looks at the
contact sheet and deletes any crop that is a real vehicle; `accept` converts
what survived. Two phases, with human eyes in between.

USAGE
-----
    # 1. propose candidates and write a reviewable contact sheet
    python mine_hard_negatives.py scan VIDEO.mp4 --out mined/ --weights best.pt

    # 2. open mined/review/ and DELETE any crop that is a real vehicle
    #    (including a legitimately stopped one)

    # 3. convert the survivors into background images + empty labels
    python mine_hard_negatives.py accept mined/ --dataset <dataset_root>
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2

# A box must sit still this long before it is even proposed. Long enough that
# genuinely slow traffic through a blind curve is not swept up.
MIN_TRACK_SECS = 2.0
# Centroid drift budget over the whole track, in pixels of the 640-wide frame.
MAX_DRIFT_PX = 12.0
# IoU for "same object, next frame".
IOU_MATCH = 0.5
# Frames kept per confirmed phantom. More than a handful just duplicates one
# scene and skews the background ratio.
FRAMES_PER_CANDIDATE = 6


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def centroid(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _prep(frame, rot, width):
    if rot is not None:
        frame = cv2.rotate(frame, rot)
    h, w = frame.shape[:2]
    return cv2.resize(frame, (width, int(h * width / w)))


def scan(args):
    from ultralytics import YOLO

    rot = {"cw": cv2.ROTATE_90_CLOCKWISE,
           "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
           "180": cv2.ROTATE_180}.get(args.rotate)

    out = Path(args.out)
    review = out / "review"
    frames_dir = out / "frames"
    for d in (review, frames_dir):
        d.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("cannot open " + args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tracks = []
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        frame = _prep(frame, rot, args.width)

        # Low threshold on purpose: a phantom that fires at 0.30 today is one
        # bad night away from firing at 0.55, so it is worth training against.
        res = model.predict(frame, conf=args.conf, verbose=False)[0]

        dets = []
        for b in res.boxes:
            dets.append((b.xyxy[0].tolist(), int(b.cls[0]), float(b.conf[0])))

        used = set()
        for xy, cls, conf in dets:
            best, best_iou = None, IOU_MATCH
            for t in tracks:
                if t["last"] < idx - 3 or id(t) in used:
                    continue
                s = iou(t["boxes"][-1], xy)
                if s >= best_iou:
                    best, best_iou = t, s
            if best is None:
                tracks.append({"boxes": [xy], "frames": [idx], "first": idx,
                               "last": idx, "cls": cls, "conf": conf})
            else:
                best["boxes"].append(xy)
                best["frames"].append(idx)
                best["last"] = idx
                best["conf"] = max(best["conf"], conf)
                used.add(id(best))

        if idx % 200 == 0:
            print("  " + str(idx) + " frames, " + str(len(tracks)) + " tracks",
                  flush=True)

    cap.release()

    names = model.names
    candidates = []
    for t in tracks:
        dur = (t["last"] - t["first"]) / fps
        if dur < MIN_TRACK_SECS:
            continue
        cs = [centroid(b) for b in t["boxes"]]
        drift = max(max(abs(c[0] - cs[0][0]) for c in cs),
                    max(abs(c[1] - cs[0][1]) for c in cs))
        if drift > MAX_DRIFT_PX:
            continue
        candidates.append({
            "cls": names.get(t["cls"], str(t["cls"])),
            "conf": round(t["conf"], 3),
            "seconds": round(dur, 1),
            "drift_px": round(drift, 1),
            "box": [round(v, 1) for v in t["boxes"][-1]],
            "frames": t["frames"],
        })

    candidates.sort(key=lambda c: -c["seconds"])

    wanted = {}
    for i, c in enumerate(candidates):
        step = max(1, len(c["frames"]) // FRAMES_PER_CANDIDATE)
        picks = c["frames"][::step][:FRAMES_PER_CANDIDATE]
        c["picked"] = picks
        for f in picks:
            wanted.setdefault(f, []).append(i)

    # Second pass writes only the frames actually referenced, so memory stays
    # flat regardless of video length.
    cap = cv2.VideoCapture(args.video)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx not in wanted:
            continue
        frame = _prep(frame, rot, args.width)
        cv2.imwrite(str(frames_dir / ("f%06d.jpg" % idx)), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        for ci in wanted[idx]:
            c = candidates[ci]
            marked = frame.copy()
            x1, y1, x2, y2 = [int(v) for v in c["box"]]
            cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 0, 255), 2)
            tag = ("#%d %s %.2f %.1fs drift%.1fpx"
                   % (ci, c["cls"], c["conf"], c["seconds"], c["drift_px"]))
            cv2.putText(marked, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(review / ("cand%03d_f%06d.jpg" % (ci, idx))), marked)
    cap.release()

    (out / "candidates.json").write_text(json.dumps(candidates, indent=2))

    print("")
    print("=== SCAN COMPLETE ===")
    print("  static candidates : " + str(len(candidates)))
    for i, c in enumerate(candidates[:20]):
        print("   #%-3d %-11s conf=%.2f %5.1fs drift=%5.1fpx box=%s"
              % (i, c["cls"], c["conf"], c["seconds"], c["drift_px"], c["box"]))
    print("")
    print("  review crops      : " + str(review))
    print("  REVIEW NOW: delete any crop showing a REAL vehicle.")
    print("  A legitimately STOPPED vehicle is real - keep the system able to")
    print("  see it. Delete those crops so they never become negatives.")
    print("  then: python mine_hard_negatives.py accept " + str(out)
          + " --dataset <root>")


def accept(args):
    out = Path(args.out)
    review, frames_dir = out / "review", out / "frames"
    cands = json.loads((out / "candidates.json").read_text())

    # A candidate survives only if at least one of its crops is still on disk.
    survived = set()
    for p in review.glob("cand*_f*.jpg"):
        survived.add(int(p.name[4:7]))

    keep_frames = set()
    for i in sorted(survived):
        keep_frames.update(cands[i]["picked"])

    ds = Path(args.dataset)
    img_dir = ds / args.split / "images"
    lbl_dir = ds / args.split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for f in sorted(keep_frames):
        src = frames_dir / ("f%06d.jpg" % f)
        if not src.exists():
            continue
        stem = "busay_neg_%s_%06d" % (args.tag, f)
        shutil.copy2(src, img_dir / (stem + ".jpg"))
        # An EMPTY label file is what makes this a background image.
        (lbl_dir / (stem + ".txt")).write_text("")
        n += 1

    print("  candidates confirmed as phantoms : %d/%d" % (len(survived), len(cands)))
    print("  background images written        : %d -> %s" % (n, img_dir))
    print("  each has an EMPTY .txt label     : %s" % lbl_dir)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="propose static-phantom candidates")
    s.add_argument("video")
    s.add_argument("--out", default="mined")
    s.add_argument("--weights", required=True)
    s.add_argument("--conf", type=float, default=0.20)
    s.add_argument("--width", type=int, default=640)
    s.add_argument("--rotate", choices=["none", "cw", "ccw", "180"],
                   default="ccw")
    s.set_defaults(func=scan)

    a = sub.add_parser("accept", help="convert survivors into background images")
    a.add_argument("out")
    a.add_argument("--dataset", required=True)
    a.add_argument("--split", default="train")
    a.add_argument("--tag", default="camA")
    a.set_defaults(func=accept)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
