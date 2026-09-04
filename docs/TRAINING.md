# Training the vehicle detection model

How the Busay vehicle model is built, why it currently produces phantom
detections, and the method for fixing them. Written 2026-09-05.

## The problem this is solving

The deployed model (`vehicle_yolo26n_20260203_032528/weights/best.pt`, 5.4 MB)
reports **mAP50 0.90 / mAP50-95 0.70**, yet in the field it:

* boxes a **guardrail post** as `truck 0.51` — above the production threshold —
  and attaches a fabricated `39 km/h` to it
* logs a motionless object at ~(555, 345) roughly **eight times a second**,
  inflating "vehicles today"

Those two numbers cannot both be right. The mAP is measured on a validation set
that does not resemble the road.

### Evidence: the video test

`testing/annotate_video.py` replays a real night clip through the exact Pi
pipeline (rotate, downscale to 640, JPEG q50, POST to the live service).
1,069 frames:

```
frames with a box: 575 (54%)
total detections : 635
by class         : truck=326, motorcycle=275, car=31, bus=3
confidence       : min=0.25 median=0.36 max=0.78
above prod 0.50  : 87/635 (14%)
```

`truck` dominates at 326 while making up only **10%** of the training boxes,
and `car` collapses to 31 despite being **23%**. The model is not slightly off
on this footage; its class posterior is close to unusable in the dark.

**This rules out lowering the threshold to 0.35.** At 0.50 it already boxes
scenery. Lowering it multiplies phantoms.

### Evidence: the phantom, isolated and identified ✅

`training/mine_hard_negatives.py scan` was run against the night clip on
irm-pc's GPU using the deployed weights. Of **87 tracks**, exactly **2** were
static enough to be flagged:

```
#0  truck       conf=0.63  6.4s  drift=6.4px  box=[0.2, 254.6, 102.6, 360.0]
#1  motorcycle  conf=0.55  2.1s  drift=5.4px  box=[44.1, 263.6,  97.1, 359.2]
```

The review crops were inspected by eye. **Both boxes land on the same fixed
object** — the concrete road-edge barrier at the bottom-left shoulder, where the
carriageway bends into the curve. It is not a vehicle in any frame.

That single piece of roadside furniture is being reported as a **truck at 0.63**
and a **motorcycle at 0.55**. Two different classes, on one static object, both
comfortably above the production threshold of 0.50.

This confirms the diagnosis and settles the remedy:

* It is **not a threshold problem.** 0.63 clears any threshold that still lets
  real vehicles through. Raising the threshold cannot separate these.
* It is **not fixable with more public data.** The barrier is unique to this
  road. Only frames of *this* shoulder can teach the model what it is.
* It **is** a background-image problem, exactly the case the literature
  describes.

### Evidence: what is wrong with the dataset

Measured on `datasets/processed/busay_vehicle_detection`:

| Finding | Value | Why it matters |
|---|---|---|
| Train images | 11,493 | — |
| **Val images** | **247 (2.1%)** | Far below the standard 10%. Too small to resolve a real difference between runs. |
| **`bicycle` instances** | **0** | A declared class the model has never seen. It can only ever be a source of spurious output. |
| Class skew | motorcycle 65,223 · car 23,869 · truck 10,757 · bus 5,403 | 6:1 imbalance head to tail. |
| `path:` in data.yaml | `C:\Projects\...\scripts\training\..\..` | Absolute path into a **different checkout on another drive**, which exists on this machine. Training could silently read the wrong data. |

The val split is also drawn from the same Roboflow sources as train, so 0.90
mAP measures *interpolation within those datasets* — not performance on a
Cebu blind curve at night. This is the textbook case of a metric that looks
finished and predicts nothing.

## What the literature says

Sources consulted:

* [Tips for Best YOLO Training Results — Ultralytics](https://docs.ultralytics.com/yolov5/tutorials/tips-for-best-training-results)
* [Downloading Background Images For YOLO Training](https://y-t-g.github.io/tutorials/bg-images-for-yolo/)
* [What does "background" in the confusion matrix mean? — ultralytics/yolov5 #9907](https://github.com/ultralytics/yolov5/discussions/9907)
* [High False Positives After Scaling YOLO Model and Dataset — ultralytics #23043](https://github.com/orgs/ultralytics/discussions/23043)
* [Training Strategy, Data Splits, and Hard Negatives for YOLOv11-Nano — ultralytics #22888](https://github.com/ultralytics/ultralytics/issues/22888)
* [Is hard negative mining used in YOLOv8? — ultralytics #3466](https://github.com/ultralytics/ultralytics/issues/3466)
* [Model Testing guide — Ultralytics](https://docs.ultralytics.com/guides/model-testing)
* [Train/Validation/Test split — Roboflow](https://blog.roboflow.com/train-test-split/)
* [Train Test Validation Split: Best Practices — Lightly](https://www.lightly.ai/blog/train-test-validation-split)
* [On the Value of Out-of-Distribution Testing (Goodhart's Law)](https://arxiv.org/pdf/2005.09241)

The four points that apply directly here:

**1. Background images are the fix for false positives.** An image with an
**empty label file** teaches the model what "nothing here" looks like.
Ultralytics' own guidance is 0–10% (COCO ships ~1%). More recent measurement
across YOLOv8-n and YOLOv11-m found FP count **minimised near 20%**, with
recall and mAP50 also peaking there — but **declining beyond it**, as too many
negatives push the model into over-conservatism and it starts missing real
objects. So 20% is a ceiling to approach, not exceed.

**2. Hard negative mining beats threshold tuning.** The recommended loop is:
run inference, collect the false positives, fold them back in as background,
retrain. Raising the confidence threshold at inference "helps short-term but
does not fix the root cause" — and here it is not even available, because the
phantoms already clear 0.50.

**3. The generic fix for domain shift is a small amount of your own
environment.** Public data cannot supply this. **No dataset on earth contains
our guardrail.** Frames from the Busay cameras are the only thing that can
teach the model that this particular post is not a truck.

**4. Val and test must be separated, and one val set should be
production-realistic.** All tuning happens on val; test is touched once. The
Ultralytics data YAML accepts a **list** of val sets, which lets a curated
regression set track regressions while a Busay-realistic set tracks domain
shift. The 0.90 figure is a live demonstration of Goodhart's law.

## The pipeline

### 1. Acquire and harmonise — `training/acquire_datasets.py`

Adds three sources to the existing Busay data:

| Key | Roboflow project | Images | Why |
|---|---|---|---|
| `ph` | [`vehicle-detection-and-data-generation/ph-vehicles-728sm`](https://universe.roboflow.com/vehicle-detection-and-data-generation/ph-vehicles-728sm) | ~4,936 | Philippine vehicles. Jeepneys, tricycles and habal-habal appear in no other public dataset and are most of Busay's traffic. |
| `thammasat` | [`thammasat-university-voac4/vehicle-detection-from-cctv-night`](https://universe.roboflow.com/thammasat-university-voac4/vehicle-detection-from-cctv-night) | ~524 | Night CCTV from a fixed pole — the exact geometry and lighting the model is currently worst at. |
| `uadetrac` | [`cs474-ug2-vehicle-detection/ua-detrac-rvwkg`](https://universe.roboflow.com/cs474-ug2-vehicle-detection/ua-detrac-rvwkg) | large | Fixed-camera surveillance benchmark; scale and viewpoint variety. |

**Requires a Roboflow API key** (free, <https://app.roboflow.com/settings/api>).
There is no anonymous programmatic download path.

#### The taxonomy trap

The sources do not share a taxonomy. Thammasat has a generic `vehicle`
alongside `car`; UA-DETRAC has `van` and `others`. There is no honest mapping
from a generic `vehicle` box to one of five specific classes.

The tempting shortcut — drop the box, keep the image — is **actively harmful**.
An unlabelled real vehicle in a training image teaches the model that a vehicle
is background. That manufactures **false negatives** on the one road that
matters, which is the failure direction that gets someone hurt.

**Policy: if an image contains any box that cannot be mapped, the whole image
is excluded.** Every exclusion is counted and printed. `--map-generic-to car`
overrides this, and is opt-in on purpose.

### 2. Mine hard negatives — `training/mine_hard_negatives.py`

This is the step that actually addresses the guardrail.

`scan` tracks detections by IoU across frames and proposes any cluster whose
centroid drifts **< 12 px over ≥ 2 s** — on a pole-mounted camera, a box that
does not move is scenery.

**It deliberately does not auto-accept.** A genuinely stopped vehicle is also
motionless, and `stopped_vehicle` is an incident this system must detect.
Auto-labelling static detections as background would train the model to ignore
exactly the hazard it exists to warn about. So `scan` writes annotated crops to
`mined/review/`, a human deletes any crop showing a real vehicle, and `accept`
converts only the survivors into background images with empty labels.

```bash
python mine_hard_negatives.py scan clip.mp4 --out mined/ --weights best.pt
#   ... review mined/review/ by eye, delete real vehicles ...
python mine_hard_negatives.py accept mined/ --dataset <dataset_root>
```

Target: negatives at **10–20%** of the training set, per the finding above.

### 3. Train

On irm-pc (RTX 3060 Ti, 8 GB). Verified present: torch 2.5.1+cu121, CUDA
available, ultralytics 8.4.123, venv at
`D:\RoadSentinel\server\ai-service\venv`.

```bash
python train.py --dataset vehicle --model-size n --epochs 100
```

### 4. Validate against the failure, not the metric

mAP on a same-distribution val set is what produced 0.90 while the model boxed
a guardrail. The acceptance test is the **Busay night clip**:

```bash
python testing/annotate_video.py clip.mp4 --out after.mp4
```

Compare against the pre-training baseline recorded above. What must improve:

* `truck` count falls sharply — 326 on a night road is not plausible
* median confidence on real vehicles rises above 0.36
* **no box on the guardrail at or above 0.50**

A higher mAP with the guardrail still boxed is a **failed** run.

## Open items

* ⚠️ **Not enough negatives yet.** The single available clip yields only **12**
  background frames from 2 confirmed phantoms — against 11,493 training images,
  that is 0.1%, far short of the 10–20% the literature calls for. The barrier
  will be under-represented and may survive retraining.
  **What is needed: more night footage of this road, especially with an empty
  carriageway.** Ten minutes of quiet-hour recording from each camera would
  supply several hundred true background frames of the exact scene. This is the
  highest-value thing a human can contribute to the next model.
* 🟡 **Unverified — needs the Roboflow API key.** No external dataset has been
  downloaded; `acquire_datasets.py` has not been run end to end.
* 🟡 **Unverified — no training run has been performed.** The dataset audit and
  the phantom identification above are measured and confirmed; nothing in the
  training or post-training sections has been executed.
* The `bicycle` class has zero instances. Either source data for it or drop it
  to 4 classes — a declared class with no examples can only emit noise.
* Server-side, one static object still logs ~8×/sec. Even a perfect model wants
  per-track deduplication before rows reach `detections`.
