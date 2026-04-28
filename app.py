"""
Neoveli Semantic Similarity Microservice (OPTIMIZED)
Flask + sentence-transformers (LIGHTWEIGHT VERSION)

Deploy on VPS or Render Free (optimized for 512MB RAM)

Install:
  pip install flask sentence-transformers==2.2.2 scikit-learn numpy
"""

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

app = Flask(__name__)

# 🟢 LIGHTWEIGHT MODEL (IMPORTANT FIX FOR MEMORY ERROR)
MODEL_NAME = os.getenv('MODEL_NAME', 'paraphrase-MiniLM-L3-v2')

print(f"Loading lightweight model: {MODEL_NAME}...")

model = SentenceTransformer(
    MODEL_NAME,
    device='cpu'  # 🔥 force CPU (avoids GPU memory crash)
)

print("Model loaded successfully.")

def get_embedding(text: str):
    """Generate embedding safely (memory optimized)."""
    text = text[:2000]  # 🔥 reduced for memory stability
    return model.encode([text], convert_to_numpy=True)[0]

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': MODEL_NAME,
        'mode': 'cpu-lightweight'
    })

@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json(force=True, silent=True)

    if not data or 'text1' not in data or 'text2' not in data:
        return jsonify({'error': 'text1 and text2 required'}), 400

    text1 = str(data['text1'])[:3000]
    text2 = str(data['text2'])[:3000]

    # 🔥 avoid useless computation
    if len(text1.split()) < 5 or len(text2.split()) < 5:
        return jsonify({'similarity': 0.0})

    emb1 = get_embedding(text1).reshape(1, -1)
    emb2 = get_embedding(text2).reshape(1, -1)

    sim = float(cosine_similarity(emb1, emb2)[0][0])

    # clamp
    sim = max(0.0, min(1.0, sim))

    return jsonify({
        'similarity': round(sim, 4)
    })

@app.route('/compare_chunks', methods=['POST'])
def compare_chunks():
    data = request.get_json(force=True, silent=True)

    if not data or 'text' not in data or 'corpus' not in data:
        return jsonify({'error': 'text and corpus required'}), 400

    text = str(data['text'])[:3000]
    corpus = [str(c)[:2000] for c in data['corpus'][:20]]  # 🔥 reduced for memory

    if not corpus:
        return jsonify({'best_similarity': 0.0, 'scores': []})

    emb_text = get_embedding(text).reshape(1, -1)
    emb_corpus = model.encode(corpus, convert_to_numpy=True)

    scores = cosine_similarity(emb_text, emb_corpus)[0].tolist()
    scores = [max(0.0, min(1.0, float(s))) for s in scores]

    return jsonify({
        'best_similarity': round(max(scores), 4),
        'scores': [round(s, 4) for s in scores]
    })

# 🟢 IMPORTANT RENDER FIX
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
