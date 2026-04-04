from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import json
import os
from tensorflow.keras.models import load_model
import anthropic

app = Flask(__name__)
CORS(app)

ANTHROPIC_API_KEY = "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"

# ── Load chest disease model ──────────────────────────────────────────────────
print("Loading chest disease model...")
chest_model = load_model("disease_model.h5")

if os.path.exists("class_indices.json"):
    with open("class_indices.json") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
else:
    idx_to_class = {0: "COVID", 1: "LUNG_OPACITY", 2: "NORMAL", 3: "PNEUMONIA", 4: "TUBERCULOSIS"}

print(f"Chest model loaded! Classes: {idx_to_class}")

# ── Load spine model ──────────────────────────────────────────────────────────
SPINE_CLASSES = ['Normal', 'Scoliosis', 'Spondylolisthesis']
spine_model = None

if os.path.exists("spine_model.h5"):
    print("Loading spine model...")
    spine_model = load_model("spine_model.h5")
    # Warm up the model — CRITICAL for Keras 3 to work properly
    _dummy = np.zeros((1, 150, 150, 3), dtype=np.float32)
    spine_model.predict(_dummy, verbose=0)
    print("Spine model loaded and ready!")
else:
    print("WARNING: spine_model.h5 not found — copy it to this folder")

print("Server is ready.")


# ── Chest X-ray prediction ────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """Chest X-ray: COVID, LUNG_OPACITY, NORMAL, PNEUMONIA, TUBERCULOSIS"""
    try:
        data = request.get_json()
        image_base64 = data["image"]

        image_bytes = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

        if image is None:
            return jsonify({"error": "Could not read image"}), 400

        image = cv2.resize(image, (150, 150))
        image = np.stack((image,) * 3, axis=-1)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = chest_model.predict(image)[0]

        top_idx = int(np.argmax(predictions))
        top_label = idx_to_class[top_idx]
        top_confidence = float(predictions[top_idx] * 100)

        all_scores = {
            idx_to_class[i]: round(float(p * 100), 1)
            for i, p in enumerate(predictions)
        }

        return jsonify({
            "label": top_label,
            "confidence": round(top_confidence, 1),
            "isNormal": top_label.upper() == "NORMAL",
            "allScores": all_scores
        })

    except Exception as e:
        print("Chest prediction error:", e)
        return jsonify({"error": str(e)}), 500


# ── Spine X-ray prediction ────────────────────────────────────────────────────
@app.route("/spine", methods=["POST"])
def spine():
    """Spine X-ray: Normal, Scoliosis, Spondylolisthesis"""
    if spine_model is None:
        return jsonify({"error": "spine_model.h5 not found. Copy it to the project folder."}), 503

    try:
        data = request.get_json()
        image_base64 = data["image"]

        # Decode image
        image_bytes  = base64.b64decode(image_base64)
        image_array  = np.frombuffer(image_bytes, dtype=np.uint8)
        image        = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Could not read image"}), 400

        # Preprocess
        image_rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (150, 150))
        image_norm    = image_resized / 255.0
        img_input     = np.expand_dims(image_norm, axis=0).astype(np.float32)

        # Predict
        predictions = spine_model.predict(img_input, verbose=0)[0]
        top_idx     = int(np.argmax(predictions))
        top_label   = SPINE_CLASSES[top_idx]
        top_conf    = round(float(predictions[top_idx] * 100), 1)

        all_scores = {
            SPINE_CLASSES[i]: round(float(predictions[i] * 100), 1)
            for i in range(len(SPINE_CLASSES))
        }

        print(f"[Spine] {top_label} ({top_conf}%)")

        return jsonify({
            "label":      top_label,
            "confidence": top_conf,
            "isNormal":   top_label == "Normal",
            "allScores":  all_scores
        })

    except Exception as e:
        print("Spine prediction error:", e)
        return jsonify({"error": str(e)}), 500


# ── Claude AI chat ────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    """Forwards chat messages to Claude AI."""
    try:
        data = request.get_json()
        print("Chat request received, message count:", len(data["messages"]))

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=data["system"],
            messages=data["messages"]
        )

        reply_text = response.content[0].text
        print("Claude replied, length:", len(reply_text))

        return jsonify({
            "content": [{"type": "text", "text": reply_text}]
        })

    except Exception as e:
        print("Chat error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
