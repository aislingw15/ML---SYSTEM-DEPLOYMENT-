from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import json
import os
import time
from tensorflow.keras.models import load_model
import anthropic

app = Flask(__name__)
CORS(app)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Medical Knowledge Base (RAG) ──────────────────────────────────────────────
KNOWLEDGE_BASE = {
    "covid": "COVID-19 on X-ray shows bilateral ground-glass opacities, often peripheral and lower lobe dominant. CT is more sensitive. Symptoms: fever, dry cough, loss of smell/taste, fatigue. Treatment: supportive care, antivirals in severe cases.",
    "pneumonia": "Pneumonia appears as consolidation (white patches) on X-ray, often with air bronchograms. Can affect one lobe (lobar) or be patchy (bronchopneumonia). Bacterial pneumonia treated with antibiotics; viral pneumonia with supportive care.",
    "tuberculosis": "TB on X-ray shows upper lobe infiltrates, cavitation, and hilar lymphadenopathy. Classic symptoms: cough >3 weeks, haemoptysis, night sweats, weight loss. Treatment: 6-month RIPE therapy (Rifampin, Isoniazid, Pyrazinamide, Ethambutol).",
    "lung opacity": "Lung opacity refers to haziness in the lung fields caused by fluid, infection, collapse, or mass. Causes include pneumonia, pulmonary oedema, haemorrhage, or tumour. Requires further investigation with CT scan.",
    "normal": "A normal chest X-ray shows clear lung fields, normal cardiac silhouette, and no infiltrates or masses. However, a normal X-ray does not rule out all lung conditions — early TB and small pneumothorax can be missed.",
    "scoliosis": "Scoliosis is a lateral curvature of the spine measured by Cobb angle. >10° is diagnostic. Mild (<20°): observe. Moderate (20-45°): bracing. Severe (>45°): surgical fusion. Symptoms: uneven shoulders, back pain, visible curve.",
    "spondylolisthesis": "Spondylolisthesis is forward slippage of one vertebra over another, graded I-IV by percentage. Grade I-II: physiotherapy, pain management. Grade III-IV: surgery. Symptoms: lower back pain, sciatica, leg weakness.",
    "spine": "Spine X-rays evaluate vertebral alignment, disc spaces, and bone structure. AP and lateral views are standard. Key measurements: Cobb angle (scoliosis), slip percentage (spondylolisthesis), disc height.",
    "xray": "X-rays use ionising radiation to image internal structures. Chest X-rays show lungs, heart, ribs. Spine X-rays show vertebrae and alignment. Low radiation dose. Digital X-rays give immediate results.",
    "back pain": "Back pain affects 80% of people at some point. Red flags requiring urgent imaging: pain after trauma, night pain, weight loss, fever, neurological symptoms. Most acute back pain resolves within 6 weeks.",
}

def retrieve_knowledge(query: str) -> str:
    """Simple keyword-based RAG retrieval from medical knowledge base."""
    query_lower = query.lower()
    results = []
    for key, value in KNOWLEDGE_BASE.items():
        if key in query_lower:
            results.append(value)
    if results:
        return " | ".join(results[:2])  # Return top 2 matches
    return ""

# ── Metrics Storage ───────────────────────────────────────────────────────────
METRICS_FILE = "metrics.json"

def load_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            return json.load(f)
    return {
        "total_requests": 0,
        "ttft_values": [],
        "tpot_values": [],
        "feedback": {"thumbs_up": 0, "thumbs_down": 0},
        "judge_scores": [],
        "sessions": []
    }

def save_metrics(m):
    with open(METRICS_FILE, "w") as f:
        json.dump(m, f, indent=2)

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
    _dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    spine_model.predict(_dummy, verbose=0)
    print("Spine model loaded and ready!")
else:
    print("WARNING: spine_model.h5 not found")

print("Server is ready.")


# ── Chest X-ray prediction ────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
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
    if spine_model is None:
        return jsonify({"error": "spine_model.h5 not found"}), 503

    try:
        data = request.get_json()
        image_base64 = data["image"]

        image_bytes  = base64.b64decode(image_base64)
        image_array  = np.frombuffer(image_bytes, dtype=np.uint8)
        image        = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Could not read image"}), 400

        image_rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (224, 224))
        image_norm    = image_resized / 255.0
        img_input     = np.expand_dims(image_norm, axis=0).astype(np.float32)

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


# ── Claude AI Chat with TTFT/TPOT tracking + RAG ─────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        messages = data.get("messages", [])
        system_prompt = data.get("system", "You are MediScan AI.")

        # ── RAG: retrieve relevant knowledge ──────────────────────────────
        last_message = messages[-1]["content"] if messages else ""
        rag_context = retrieve_knowledge(last_message)
        if rag_context:
            system_prompt += f"\n\nRelevant medical knowledge:\n{rag_context}"
            print(f"[RAG] Retrieved knowledge for: {last_message[:50]}...")

        print("Chat request received, message count:", len(messages))

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # ── TTFT tracking ─────────────────────────────────────────────────
        request_start = time.time()

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            system=system_prompt,
            messages=messages
        )

        first_token_time = time.time()
        reply_text = response.content[0].text
        end_time = time.time()

        # ── Calculate TTFT and TPOT ───────────────────────────────────────
        ttft_ms = round((first_token_time - request_start) * 1000, 2)
        token_count = len(reply_text.split())
        total_time = end_time - first_token_time
        tpot_ms = round((total_time / max(token_count, 1)) * 1000, 2)

        # ── Log metrics ───────────────────────────────────────────────────
        metrics = load_metrics()
        metrics["total_requests"] += 1
        metrics["ttft_values"].append(ttft_ms)
        metrics["tpot_values"].append(tpot_ms)
        save_metrics(metrics)

        print(f"Claude replied. TTFT: {ttft_ms}ms | TPOT: {tpot_ms}ms | Tokens: {token_count}")

        return jsonify({
            "content": [{"type": "text", "text": reply_text}],
            "rag_used": bool(rag_context),
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms
        })

    except Exception as e:
        print("Chat error:", e)
        return jsonify({"error": str(e)}), 500


# ── Human Feedback ────────────────────────────────────────────────────────────
@app.route("/feedback", methods=["POST"])
def feedback():
    """Record thumbs up/down human feedback."""
    try:
        data = request.get_json()
        feedback_type = data.get("type")  # 'up' or 'down'
        query = data.get("query", "")

        metrics = load_metrics()
        if feedback_type == "up":
            metrics["feedback"]["thumbs_up"] += 1
        elif feedback_type == "down":
            metrics["feedback"]["thumbs_down"] += 1

        metrics["sessions"].append({
            "type": feedback_type,
            "query": query[:100]
        })
        save_metrics(metrics)

        total = metrics["feedback"]["thumbs_up"] + metrics["feedback"]["thumbs_down"]
        satisfaction = round((metrics["feedback"]["thumbs_up"] / max(total, 1)) * 100, 1)

        print(f"[Feedback] {feedback_type} | Satisfaction: {satisfaction}%")

        return jsonify({
            "status": "recorded",
            "thumbs_up": metrics["feedback"]["thumbs_up"],
            "thumbs_down": metrics["feedback"]["thumbs_down"],
            "satisfaction_rate": satisfaction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── AI-as-Judge ───────────────────────────────────────────────────────────────
@app.route("/evaluate", methods=["POST"])
def evaluate():
    """Claude grades its own response using custom rubric."""
    try:
        data = request.get_json()
        user_query = data.get("query", "")
        ai_response = data.get("response", "")
        scan_result = data.get("scan_result", None)

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        judge_prompt = f"""You are a medical AI quality evaluator. Score this AI response.

User Query: {user_query}
Scan Result: {json.dumps(scan_result) if scan_result else 'None'}
AI Response: {ai_response}

Score each criterion 1-10 and respond ONLY with valid JSON:
{{
  "medical_accuracy": <1-10>,
  "clarity": <1-10>,
  "safety": <1-10>,
  "relevance": <1-10>,
  "overall": <1-10>,
  "reasoning": "<one sentence>"
}}"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": judge_prompt}]
        )

        result_text = response.content[0].text.strip()
        # Clean JSON
        if "```" in result_text:
            result_text = result_text.split("```")[1].replace("json", "").strip()

        scores = json.loads(result_text)

        # Log judge scores
        metrics = load_metrics()
        metrics["judge_scores"].append(scores)
        save_metrics(metrics)

        print(f"[AI-Judge] Overall score: {scores.get('overall', 0)}/10")

        return jsonify(scores)

    except Exception as e:
        print("Evaluate error:", e)
        return jsonify({"error": str(e)}), 500


# ── Metrics Dashboard ─────────────────────────────────────────────────────────
@app.route("/metrics", methods=["GET"])
def metrics_dashboard():
    """Return all evaluation metrics."""
    try:
        m = load_metrics()

        ttft = m["ttft_values"]
        tpot = m["tpot_values"]
        judge = [s.get("overall", 0) for s in m["judge_scores"]]
        total_fb = m["feedback"]["thumbs_up"] + m["feedback"]["thumbs_down"]

        return jsonify({
            "total_requests":   m["total_requests"],
            "avg_ttft_ms":      round(sum(ttft) / max(len(ttft), 1), 2),
            "avg_tpot_ms":      round(sum(tpot) / max(len(tpot), 1), 2),
            "avg_judge_score":  round(sum(judge) / max(len(judge), 1), 2),
            "thumbs_up":        m["feedback"]["thumbs_up"],
            "thumbs_down":      m["feedback"]["thumbs_down"],
            "satisfaction_pct": round((m["feedback"]["thumbs_up"] / max(total_fb, 1)) * 100, 1),
            "total_evaluations": len(judge),
            "rag_enabled":      True,
            "knowledge_entries": len(KNOWLEDGE_BASE)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
