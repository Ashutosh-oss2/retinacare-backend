# RetinaCare AI Backend (`retinacare-backend`)

> **Notice**: *This repository contains the standalone **Backend and Machine Learning Inference System** for the RetinaCare platform. The frontend web application is maintained in a separate repository.*

> **Clinical Regulatory Disclaimer**:  
> *"This system provides AI-assisted retinal screening and segmentation support and is not a substitute for professional medical diagnosis. All classification grades, vascular tracings, and lesion overlay masks must be verified by a certified ophthalmologist or qualified medical professional prior to clinical intervention."*

---

## 1. Project Overview

`retinacare-backend` is a production-grade, modular FastAPI microservice that provides automated, AI-assisted retinal analysis from standard fundus photographs.

It orchestrates three specialized deep learning models:
1. **Model 1: APTOS Diabetic Retinopathy Classification (`EfficientNetB0`)** — 5-class DR severity staging and referable DR triage with automated image quality assessment.
2. **Model 2: DRIVE Retinal Vessel Segmentation (`3-Level PyTorch U-Net`)** — Pixel-level retinal vasculature segmentation and vascular density computation.
3. **Model 3: IDRiD Retinal Lesion Segmentation (`4-Channel PyTorch U-Net`)** — Segmentation of 4 distinct DR lesion types (Microaneurysms, Hemorrhages, Hard Exudates, Soft Exudates) with diagnostic color overlays.

---

## 2. Three-Model Deep Learning Architecture

```
                                  Retinal Fundus Image (.jpg / .png / .tif)
                                                    │
                             ┌──────────────────────┴──────────────────────┐
                             │                                             │
               ┌─────────────▼─────────────┐                 ┌─────────────▼─────────────┐
               │    Quality Assessment     │                 │   Green Channel + CLAHE   │
               │   (Focus, Illumination,   │                 │     Adaptive Contrast     │
               │      Field of View)       │                 └─────────────┬─────────────┘
               └─────────────┬─────────────┘                               │
                             │                               ┌─────────────┴─────────────┐
               ┌─────────────▼─────────────┐                 │                           │
               │   Model 1: EfficientNetB0 │       ┌─────────▼─────────┐       ┌─────────▼─────────┐
               │   APTOS DR Classifier     │       │ Model 2: U-Net    │       │ Model 3: U-Net    │
               │  (224x224 RGB Fundus)     │       │ DRIVE Vasculature │       │ IDRiD 4-Lesion    │
               └─────────────┬─────────────┘       │ (48x48 Patches)   │       │ (256x256 RGB)     │
                             │                     └─────────┬─────────┘       └─────────┬─────────┘
                             │                               │                           │
                             ▼                               ▼                           ▼
                     [DR Grade & Risk]              [Binary Vessel Mask]         [Color Lesion Mask]
                     Referable: Yes/No               Vascular Density %          MA / HE / EX / SE
```

---

## 3. Audited Quantitative Performance Benchmarks

All performance metrics have been mathematically verified against ground-truth benchmark splits:

| Model | Architecture | Benchmark Dataset | Primary Accuracy | Sensitivity | Specificity | Precision | F1 / Dice |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Model 1: APTOS** | EfficientNetB0 | APTOS 2019 ($N=366$) | **91.80%** *(Referable)*<br>**76.23%** *(5-Class)* | **93.43%** | **90.83%** | **85.91%** | **89.51%** |
| **Model 2: DRIVE** | 3-Level U-Net | DRIVE ($1,600$ patches) | **95.35%** *(Pixel Acc)* | **77.65%** | **97.71%** | **81.91%** | **79.72%** *(Dice)* |
| **Model 3: IDRiD** | 4-Channel U-Net | IDRiD Validation Split | **98.86%** *(Pixel Acc)* | **59.44%** | **98.93%** | **50.93%** | **34.35%** *(Macro Dice)* |

### Model 3 Active Checkpoint (`idrid_best_model.pth` - Epoch 2) Per-Lesion Dice:
- **Hemorrhages (HE)**: **74.21%** Dice, **91.89%** Sensitivity, **62.24%** Precision
- **Soft Exudates (SE / Cotton Wool)**: **51.82%** Dice, **100.00%** Sensitivity, **34.98%** Precision
- **Hard Exudates (EX)**: **11.38%** Dice, **45.88%** Sensitivity, **98.41%** Specificity
- **Microaneurysms (MA)**: **0.00%** Dice *(Extreme $0.12\%$ pixel sparsity; requires high-resolution patch magnification $>1024\times1024$)*

---

## 4. Repository Structure

```
retinacare-backend/
├── app/
│   ├── main.py                     # Application entry point, lifespan, CORS, and routing
│   ├── api/
│   │   ├── __init__.py             # Router aggregator
│   │   ├── health.py               # Health check and root discovery endpoints
│   │   ├── aptos.py                # Model 1 DR classification routes
│   │   ├── vessel.py               # Model 2 vessel segmentation routes
│   │   ├── idrid.py                # Model 3 lesion segmentation routes
│   │   ├── unified.py              # Single-endpoint multi-model analysis route
│   │   └── model_info.py           # Model architectures and audited benchmark metadata
│   ├── models/
│   │   ├── __init__.py             # Model package exports
│   │   ├── aptos_model.py          # Keras EfficientNetB0 loader
│   │   ├── vessel_model.py         # PyTorch DriveUNet definition
│   │   └── idrid_model.py          # PyTorch IdridUNet definition
│   ├── services/
│   │   ├── __init__.py             # Service package exports
│   │   ├── model_registry.py       # Singleton loader holding all models in memory
│   │   ├── aptos_service.py        # APTOS inference engine
│   │   ├── vessel_service.py       # DRIVE vessel segmentation engine
│   │   └── idrid_service.py        # IDRiD lesion segmentation engine
│   ├── schemas/
│   │   ├── __init__.py             # Pydantic validation schemas
│   │   ├── health.py               # Health and discovery schemas
│   │   ├── aptos.py                # APTOS response schemas
│   │   ├── vessel.py               # Vessel response schemas
│   │   ├── idrid.py                # IDRiD response schemas
│   │   └── unified.py              # Multi-model composite schema
│   └── utils/
│       ├── __init__.py
│       ├── logging.py              # Structured application logger
│       ├── image_processing.py     # Image decoding, validation, base64 encoder
│       └── image_quality/          # Standalone quality assessment & enhancement module
├── weights/
│   ├── best_aptos_model.keras      # Model 1 pre-trained weights (16.33 MB)
│   ├── vessel_unet.pth             # Model 2 pre-trained weights (7.40 MB)
│   ├── idrid_best_model.pth        # Model 3 pre-trained weights (49.52 MB)
│   └── aptos_model_config.json     # Model 1 configuration and threshold metadata
├── evaluation/
│   ├── model_metrics.json          # Machine-readable benchmark numbers
│   ├── quantitative_evaluation_report.md
│   ├── quantitative_evaluation_audit.md
│   └── deployment_readiness_report.md
├── scripts/
│   └── run_evaluation.py           # Reproduce audited metric summaries
├── tests/
│   └── test_backend.py             # 10-point automated test suite
├── Dockerfile                      # Production container definition
├── .dockerignore                   # Docker build exclusions
├── .gitignore                      # Git safety exclusions
├── .env.example                    # Environment variables template
├── render.yaml                     # Cloud infrastructure deployment manifest
└── requirements.txt                # Pinned backend dependencies
```

---

## 5. Installation & Local Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-username/retinacare-backend.git
cd retinacare-backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Environment Variables Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env`:
```env
PORT=8000
MODEL_WEIGHTS_DIR=./weights
KERAS_BACKEND=torch
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000,https://your-frontend.vercel.app
ALLOW_ORIGIN_REGEX=https://.*\.vercel\.app
```

### 3. Run Automated Tests (100% Pass Rate)
```bash
pytest tests/test_backend.py -v
```

### 4. Start Local Development Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Base: `http://localhost:8000`
- Interactive OpenAPI / Swagger Docs: `http://localhost:8000/docs`

---

## 6. API Endpoints & Usage

### Endpoint Directory

| HTTP Method | Route Path | Description | Backward Compatibility Alias |
| :---: | :--- | :--- | :--- |
| `GET` | `/` | Root service discovery | — |
| `GET` | `/health` | Application health and model initialization status | `/api/health` |
| `GET` | `/api/model-info` | Architecture specifications and metadata | — |
| `GET` | `/api/model-stats` | Full audited quantitative metrics payload | — |
| `POST` | `/predict/aptos` | Model 1: APTOS DR severity classification + image quality gating | `/api/predict/aptos`, `/api/predict` |
| `POST` | `/predict/vessel` | Model 2: DRIVE retinal vessel segmentation + vascular density | `/api/predict/vessel`, `/predict/drive`, `/api/predict/drive` |
| `POST` | `/predict/idrid` | Model 3: IDRiD multi-lesion segmentation (MA, HE, EX, SE) | `/api/predict/idrid` |
| `POST` | `/predict/all` | Unified single-call execution of all 3 models | `/api/predict/all` |

---

### Example cURL Requests

#### 1. Model 1: APTOS DR Classification
```bash
curl -X POST http://localhost:8000/predict/aptos \
  -F "file=@sample_fundus.png"
```

#### 2. Model 2: DRIVE Vessel Segmentation
```bash
curl -X POST "http://localhost:8000/predict/vessel?threshold=0.5" \
  -F "file=@sample_fundus.png"
```

#### 3. Model 3: IDRiD Lesion Segmentation
```bash
curl -X POST "http://localhost:8000/predict/idrid?threshold=0.5" \
  -F "file=@sample_fundus.png"
```

#### 4. Unified Multi-Model Prediction (All 3 Models)
```bash
curl -X POST http://localhost:8000/predict/all \
  -F "file=@sample_fundus.png"
```

---

## 7. Production Deployment Instructions

### A. Deploy via Docker
```bash
docker build -t retinacare-backend:latest .
docker run -p 8000:8000 -e PORT=8000 retinacare-backend:latest
```

### B. Deploy on Render
This repository includes a `render.yaml` specification.
1. Connect this repository to your Render account.
2. Render will automatically detect `render.yaml` and configure the web service.
3. Add your production frontend domain to `ALLOWED_ORIGINS` under Render Environment Settings.

---

## 8. Frontend Integration

In your React / Vite frontend repository:
1. Set the backend URL in `.env.production`:
   ```env
   VITE_API_URL=https://retinacare-backend.onrender.com
   ```
2. API calls can target standard routes (`/predict/aptos`, `/predict/vessel`, `/predict/idrid`, `/predict/all`) or existing aliases (`/api/predict`).
3. Binary masks and color overlays are returned as standard base64 PNG data URLs (`data:image/png;base64,...`) for direct rendering in standard `<img>` elements without file system latency.
