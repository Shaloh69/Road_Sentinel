# Models Folder

Trained model weights for Road Sentinel, produced by `training/train.py`.

## Structure

```
models/
└── runs/
    └── <dataset>/                          # "vehicle" or "accident"
        └── <dataset>_yolo26<size>_<timestamp>/
            ├── weights/
            │   ├── best.pt                 # Best checkpoint by validation metric
            │   ├── last.pt                 # Most recent checkpoint
            │   └── epoch0.pt, epoch10.pt, ...
            ├── args.yaml                   # Exact training config used
            ├── results.csv, results.png    # Training/validation curves
            ├── confusion_matrix.png
            └── F1_curve.png, PR_curve.png, P_curve.png, R_curve.png
```

There is no `models/v1/`, `models/v2/`, or `models/production/` symlink layout — `train.py` writes directly to `models/runs/<dataset>/<run-name>/`, and the run name already encodes the dataset, model size, and timestamp, so there's no separate versioning scheme to manage by hand.

## Current state

- **Vehicle detection**: trained. `models/runs/vehicle/vehicle_yolo26n_20260203_032528/weights/best.pt` — 5 classes (car, motorcycle, bicycle, bus, truck), loaded by `server/ai-service`'s `TRAFFIC_MODEL_PATH`.
- **Crash/incident detection**: **not yet trained**. The dataset (`datasets/processed/busay_accident_detection/`) is ready; `python training/train.py --dataset accident --model-size n --epochs 100` is the command to run when you're ready to spend the GPU hours. Until then, `server/ai-service`'s incident detector runs a heuristic (brightness-variance) fallback and labels every result `isHeuristic: true` so it's never mistaken for a real detection.

## Which weight is the AI service actually using?

Check `server/ai-service/.env`'s `TRAFFIC_MODEL_PATH` — it resolves relative to `server/ai-service/`, not your shell's working directory. At startup (or on first detection call — model loading is lazy), the AI service logs `Traffic detector ready — custom_model=True/False`; `False` means it silently fell back to the stock, untrained `yolov8n.pt` instead of your weight. `GET /api/stats` also reports live load state for both models.

## Re-running or comparing training

Each run gets its own timestamped folder, so re-running `train.py` never overwrites a previous run — just point `TRAFFIC_MODEL_PATH` at whichever `weights/best.pt` you want to deploy. `training/validate.py --model <path> --source <video/image>` runs inference against a specific weight file directly, no server required, useful for comparing runs before switching the deployed one.
