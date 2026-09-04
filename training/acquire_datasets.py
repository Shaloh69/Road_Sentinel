#!/usr/bin/env python3
"""
Download and harmonise the external vehicle datasets onto Road Sentinel's
five classes, then merge them with the existing Busay set.

TARGET TAXONOMY (must match datasets/processed/busay_vehicle_detection):
    0 car   1 motorcycle   2 bicycle   3 bus   4 truck

SOURCES
-------
  ph        vehicle-detection-and-data-generation/ph-vehicles-728sm  (~4,936)
            Philippine road vehicles. The closest public match to Busay:
            jeepneys, tricycles and habal-habal appear in no other dataset.
  thammasat thammasat-university-voac4/vehicle-detection-from-cctv-night (~524)
            Night CCTV from a fixed pole - the exact geometry and lighting the
            Busay cameras see, and the condition the current model is worst at.
  uadetrac  cs474-ug2-vehicle-detection/ua-detrac-rvwkg
            Large fixed-camera surveillance benchmark; adds scale and viewpoint
            variety.

THE MAPPING TRAP, AND THE POLICY THIS SCRIPT TAKES
--------------------------------------------------
These datasets do not share a taxonomy. Thammasat carries a generic `vehicle`
class alongside `car`; UA-DETRAC has `van` and `others`. There is no honest way
to fold a generic `vehicle` box into one of five specific classes.

The tempting shortcut - drop the box and keep the image - is ACTIVELY HARMFUL.
An unlabelled real vehicle sitting in a training image teaches the model that
this thing is background, which is the opposite of what this system needs. It
would manufacture false negatives on the one road that matters.

So the policy is: if an image contains ANY box whose class cannot be mapped,
the WHOLE IMAGE is excluded. Losing an image costs a little data; mislabelling
one costs accuracy in the direction that gets people hurt on a blind curve.
Every exclusion is counted and reported so the cost is visible, never silent.

`--map-generic-to car` overrides this for the generic-vehicle case if the count
turns out to be large enough to matter. It is opt-in on purpose.

USAGE
-----
    python acquire_datasets.py --api-key XXXX --out D:/RoadSentinel/datasets/raw
    python acquire_datasets.py --merge-only --out D:/RoadSentinel/datasets/raw \
        --busay D:/RoadSentinel/datasets/processed/busay_vehicle_detection \
        --dest  D:/RoadSentinel/datasets/processed/busay_vehicle_v2
"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

CLASSES = ["car", "motorcycle", "bicycle", "bus", "truck"]
IDX = {n: i for i, n in enumerate(CLASSES)}

# Source class name (lowercased) -> target class name, or None to mark the
# name as "known but unmappable", which excludes its image under the policy
# documented above.
NAME_MAP = {
    # cars
    "car": "car", "cars": "car", "sedan": "car", "auto": "car",
    "automobile": "car", "taxi": "car", "suv": "car",
    # motorcycles - Philippine sources add local body types
    "motorcycle": "motorcycle", "motorbike": "motorcycle",
    "motor cycle": "motorcycle", "motor": "motorcycle", "moto": "motorcycle",
    "scooter": "motorcycle", "tricycle": "motorcycle",
    "habal-habal": "motorcycle", "habal habal": "motorcycle",
    # bicycles
    "bicycle": "bicycle", "bike": "bicycle", "cycle": "bicycle",
    # buses
    "bus": "bus", "buses": "bus", "minibus": "bus", "coach": "bus",
    # trucks and heavy vehicles
    "truck": "truck", "trucks": "truck", "lorry": "truck", "van": "truck",
    "heavy vehicle": "truck", "heavy-vehicle": "truck", "pickup": "truck",
    "trailer": "truck", "jeepney": "truck", "jeep": "truck",
    # known but unmappable -> excludes the image
    "vehicle": None, "others": None, "other": None, "person": None,
    "pedestrian": None, "unknown": None,
}

DATASETS = {
    "ph": ("vehicle-detection-and-data-generation", "ph-vehicles-728sm", None),
    "thammasat": ("thammasat-university-voac4",
                  "vehicle-detection-from-cctv-night", None),
    "uadetrac": ("cs474-ug2-vehicle-detection", "ua-detrac-rvwkg", 2),
}


def download(api_key, out):
    """Fetch each dataset through Roboflow's REST export endpoint.

    Deliberately NOT the `roboflow` pip package. This runs on irm-pc inside the
    ai-service venv, which is serving live traffic; that package pulls its own
    numpy / opencv / pillow pins and could shift them under the running
    detector. `requests` is already installed, and the REST endpoint returns a
    signed zip link, so nothing new is added to a production environment.
    """
    import io
    import zipfile

    import requests

    out.mkdir(parents=True, exist_ok=True)
    got = {}
    for key, (ws, proj, ver) in DATASETS.items():
        dest = out / key
        if (dest / "data.yaml").exists():
            print("  " + key + ": already present, skipping download")
            got[key] = dest
            continue

        version = ver
        if version is None:
            meta = requests.get(
                "https://api.roboflow.com/" + ws + "/" + proj,
                params={"api_key": api_key}, timeout=60)
            meta.raise_for_status()
            versions = meta.json().get("versions", [])
            if not versions:
                print("  " + key + ": no versions exposed, skipping")
                continue
            version = str(versions[0]["id"]).split("/")[-1]

        url = ("https://api.roboflow.com/" + ws + "/" + proj + "/"
               + str(version) + "/yolov8")
        print("  " + key + ": requesting export (" + ws + "/" + proj
              + " v" + str(version) + ") ...")
        r = requests.get(url, params={"api_key": api_key}, timeout=180)
        if r.status_code != 200:
            print("  " + key + ": HTTP " + str(r.status_code) + " - "
                  + r.text[:200])
            continue
        link = r.json().get("export", {}).get("link")
        if not link:
            print("  " + key + ": no export link in response")
            continue

        print("  " + key + ": downloading zip ...")
        z = requests.get(link, timeout=1800)
        z.raise_for_status()
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
            zf.extractall(dest)
        got[key] = dest
        print("  " + key + " -> " + str(dest) + "  ("
              + str(round(len(z.content) / 1e6, 1)) + " MB)")
    return got


def load_names(root):
    y = yaml.safe_load((root / "data.yaml").read_text())
    names = y.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(n) for n in names]


def harmonise(root, names, stats, map_generic):
    """Yield (image_path, remapped_label_lines) for every usable image."""
    lut = {}
    for i, n in enumerate(names):
        key = n.strip().lower()
        tgt = NAME_MAP.get(key, "UNKNOWN")
        if tgt is None and map_generic and key in ("vehicle", "others", "other"):
            tgt = map_generic
        lut[i] = tgt

    unknown = sorted({names[i] for i, t in lut.items() if t == "UNKNOWN"})
    if unknown:
        stats["unrecognised_names"].update(unknown)

    for split in ("train", "valid", "val", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            lines = []
            drop = False
            if lbl.exists():
                for ln in lbl.read_text().split("\n"):
                    ln = ln.strip()
                    if not ln:
                        continue
                    parts = ln.split()
                    try:
                        ci = int(parts[0])
                    except ValueError:
                        continue
                    tgt = lut.get(ci, "UNKNOWN")
                    if tgt is None or tgt == "UNKNOWN":
                        # Policy: one unmappable box disqualifies the image.
                        drop = True
                        stats["dropped_boxes"][names[ci] if ci < len(names)
                                               else str(ci)] += 1
                        break
                    lines.append(str(IDX[tgt]) + " " + " ".join(parts[1:]))
                    stats["kept_boxes"][tgt] += 1
            if drop:
                stats["dropped_images"] += 1
                continue
            stats["kept_images"] += 1
            yield img, lines


def copy_split(pairs, dest, split, prefix, stats):
    img_dir = dest / split / "images"
    lbl_dir = dest / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for img, lines in pairs:
        stem = prefix + "_" + img.stem
        shutil.copy2(img, img_dir / (stem + img.suffix.lower()))
        (lbl_dir / (stem + ".txt")).write_text("\n".join(lines))
        stats["written"][split] += 1


def merge(args):
    raw = Path(args.out)
    dest = Path(args.dest)
    if dest.exists() and args.clean:
        shutil.rmtree(dest)

    rng = random.Random(args.seed)
    stats = {
        "kept_images": 0, "dropped_images": 0,
        "kept_boxes": Counter(), "dropped_boxes": Counter(),
        "unrecognised_names": set(), "written": Counter(),
    }

    # ── External sources ──────────────────────────────────────────────────
    for key in DATASETS:
        root = raw / key
        if not (root / "data.yaml").exists():
            print("  " + key + ": NOT PRESENT, skipping")
            continue
        names = load_names(root)
        print("  " + key + " classes: " + ", ".join(names))
        items = list(harmonise(root, names, stats, args.map_generic_to))
        rng.shuffle(items)
        # 80/10/10, assigned here rather than trusting each source's own split
        # so every split draws from every source.
        n = len(items)
        a, b = int(n * 0.8), int(n * 0.9)
        copy_split(items[:a], dest, "train", key, stats)
        copy_split(items[a:b], dest, "valid", key, stats)
        copy_split(items[b:], dest, "test", key, stats)
        print("  " + key + ": " + str(n) + " usable images")

    # ── The existing Busay set, copied through unchanged ──────────────────
    busay = Path(args.busay)
    if busay.exists():
        for split in ("train", "valid", "test"):
            src_i = busay / split / "images"
            src_l = busay / split / "labels"
            if not src_i.is_dir():
                continue
            pairs = []
            for img in sorted(src_i.iterdir()):
                if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                lbl = src_l / (img.stem + ".txt")
                txt = lbl.read_text() if lbl.exists() else ""
                pairs.append((img, [l for l in txt.split("\n") if l.strip()]))
            copy_split(pairs, dest, split, "busay", stats)
            print("  busay/" + split + ": " + str(len(pairs)) + " images")

    write_yaml(dest)
    report(stats, dest)


def write_yaml(dest):
    # Relative `path` on purpose. The previous data.yaml hardcoded
    # C:\Projects\Thesis\...\scripts\training\..\.. which silently pointed at a
    # DIFFERENT checkout of this repo on another drive.
    y = {
        "path": str(dest.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    (dest / "data.yaml").write_text(yaml.safe_dump(y, sort_keys=False))


def report(stats, dest):
    print("")
    print("=== MERGE REPORT ===")
    print("  images kept    : " + str(stats["kept_images"]))
    print("  images dropped : " + str(stats["dropped_images"])
          + "   (contained an unmappable box)")
    if stats["dropped_boxes"]:
        print("  dropped because of these source classes:")
        for k, v in stats["dropped_boxes"].most_common():
            print("      " + k + ": " + str(v))
    if stats["unrecognised_names"]:
        print("  !! CLASS NAMES NOT IN NAME_MAP - add them and re-run:")
        for n in sorted(stats["unrecognised_names"]):
            print("      " + n)
    print("  boxes kept per target class:")
    for c in CLASSES:
        print("      " + c + ": " + str(stats["kept_boxes"][c]))
    print("  written per split:")
    for s in ("train", "valid", "test"):
        print("      " + s + ": " + str(stats["written"][s]))
    print("  data.yaml -> " + str(dest / "data.yaml"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key", help="Roboflow API key (required to download)")
    ap.add_argument("--out", default="datasets/raw",
                    help="where the raw downloads live")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--busay",
                    default="datasets/processed/busay_vehicle_detection")
    ap.add_argument("--dest", default="datasets/processed/busay_vehicle_v2")
    ap.add_argument("--map-generic-to", default=None, choices=CLASSES,
                    help="fold generic 'vehicle'/'others' into this class "
                         "instead of excluding the image (opt-in)")
    ap.add_argument("--clean", action="store_true",
                    help="delete the destination first")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if not args.merge_only:
        if not args.api_key:
            raise SystemExit(
                "--api-key is required to download.\n"
                "Get a free key at https://app.roboflow.com/settings/api\n"
                "Already downloaded? Re-run with --merge-only.")
        download(args.api_key, Path(args.out))

    merge(args)


if __name__ == "__main__":
    main()
