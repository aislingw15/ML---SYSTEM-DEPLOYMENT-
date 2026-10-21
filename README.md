# MediScan AI — ML System Deployment

An X-ray assistant that classifies chest and spine X-rays and answers questions about the results with Claude.
Built for the ML System Deployment module.

## Components
| File | What it is |
|---|---|
| `server.py` | Flask API on `localhost:5000` |
| `xray_chatbot.html` | Web front end: upload a chest or spine X-ray, see the prediction, chat about it |
| `app.py` | Earlier PyQt6 desktop app for pneumonia detection (supports DICOM) |
| `retrain.py` | Trains the 5-class chest model (MobileNetV2 transfer learning) |
| `disease_detection_multiclass.ipynb` | Multi-class chest X-ray model notebook |
| `pneumonia_detection.ipynb` | Original pneumonia detection notebook |
| `spine_retrain_efficientnet.ipynb` | Spine model: EfficientNetB3, classes Normal / Scoliosis / Spondylolisthesis |
| `disease_model.h5`, `disease_model_best.h5`, `class_indices.json` | Trained chest model (COVID, Lung Opacity, Normal, Pneumonia, Tuberculosis) |

## API (`server.py`)
| Route | Purpose |
|---|---|
| `POST /predict` | Chest X-ray classification |
| `POST /spine` | Spine X-ray classification (needs `spine_model.h5`, produced by the spine notebook) |
| `POST /chat` | Claude chat with keyword-based RAG over a medical knowledge base; tracks TTFT/TPOT latency |
| `POST /feedback` | Thumbs up/down human feedback |
| `POST /evaluate` | AI-as-judge evaluation of answers |
| `GET /metrics` | Collected latency and feedback metrics |

## Running
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python server.py
```
Then open `xray_chatbot.html` in a browser.

The API key is read from the `ANTHROPIC_API_KEY` environment variable; never commit it to the code.

> Not a medical device: for coursework and demonstration only.
