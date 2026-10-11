# SentinelVision

Visual Anomaly Detection & Defect Localization for Smart Manufacturing.

SentinelVision is a full-stack anomaly inspection system built around a PatchCore-inspired pipeline for MVTec AD industrial imagery. It combines a WideResNet-50-2 feature extractor, a FAISS-backed memory bank, and a FastAPI-based backend for production-style inspection workflows. The frontend provides a dashboard, category-aware evaluation view, and image inspection experience for real defect detection and heatmap localization.

## Features

- PatchCore-style anomaly scoring and defect localization
- WideResNet-50-2 feature extraction with FAISS nearest-neighbor search
- MVTec AD category support for 15 product categories
- REST API for health checks and image inspection
- Defect heatmap generation for localized anomaly areas
- React + Vite dashboard for evaluation and inspection workflows
- Threshold-based anomaly decisioning and severity recommendations

## Supported Categories

- bottle
- cable
- capsule
- carpet
- grid
- hazelnut
- leather
- metal_nut
- pill
- screw
- tile
- toothbrush
- transistor
- wood
- zipper

## Project Structure

- `api.py` — FastAPI application entry point
- `src/` — model, backbone, localization, feature extraction, and evaluation logic
- `results/` — trained metrics, thresholds, and generated heatmaps
- `models/` — category memory banks and category classifier
- `data/` — MVTec AD dataset samples
- `frontend/` — React/Vite dashboard and inspection UI
- `evaluate.py` and `evaluate_localization.py` — evaluation scripts
- `inference.py` — inference helpers and utilities

## Environment Setup

This project uses Python 3.11 and a local virtual environment at `.venv/`.

```bash
cd ~/Desktop/SentinelVision
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Backend Startup

```bash
cd ~/Desktop/SentinelVision
source .venv/bin/activate
uvicorn api:app --host 127.0.0.1 --port 8000
```

The API exposes:

- `GET /api/health` — backend status and device information
- `GET /api/evaluation` — per-category evaluation metrics
- `POST /api/inspect` — image upload and anomaly inspection

Example health check:

```bash
curl http://127.0.0.1:8000/api/health
```

Example inspection request:

```bash
curl -X POST 'http://127.0.0.1:8000/api/inspect' \
  -F 'category=bottle' \
  -F 'file=@data/bottle/test/broken_small/002.png'
```

## Frontend Startup

```bash
cd ~/Desktop/SentinelVision/frontend
npm install
npm run dev -- --host 127.0.0.1
```

Open the app in the browser:

- `http://127.0.0.1:5173/`

## Validation Notes

The application was verified in the local environment with:

- successful backend startup via Uvicorn
- live health endpoint response from `/api/health`
- successful inspection inference using a real defect image from `data/bottle/test/broken_small/002.png`
- successful frontend build with `npm run build`
- working browser-driven inspection flow showing a real `ANOMALY` result and defect metadata

## Troubleshooting

- If a category memory bank is missing, the backend returns a `404` for that category.
- Ensure the dataset and model files exist under `data/` and `models/` before running inspection.
- Confirm that the frontend is pointed to the API on `http://127.0.0.1:8000`.
- If the browser cannot access the camera, allow permission for webcam capture or use file upload.

## License and Usage

This project is intended for industrial visual inspection research and prototyping. It is designed to run locally and evaluate real anomaly detection workflows using the MVTec AD dataset and PatchCore-based inference.

