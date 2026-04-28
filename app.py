"""
Neoveli Semantic Similarity Microservice
Flask + sentence-transformers (all-MiniLM-L6-v2)

Deploy on a VPS:
  pip install flask sentence-transformers torch
  python app.py

Or with gunicorn:
  gunicorn -w 2 -b 0.0.0.0:5001 app:app
"""
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

app = Flask(__name__)

# Load model once at startup (~80MB, free, local)
MODEL_NAME = os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2')
print(f"Loading model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")

def get_embedding(text: str):
    """Generate embedding for a text chunk."""
    return model.encode([text[:3000]], convert_to_numpy=True)[0]

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': MODEL_NAME})

@app.route('/compare', methods=['POST'])
def compare():
    """
    Compare two texts semantically.
    Input:  { "text1": "...", "text2": "..." }
    Output: { "similarity": 0.87 }
    """
    data = request.get_json(force=True, silent=True)
    if not data or 'text1' not in data or 'text2' not in data:
        return jsonify({'error': 'text1 and text2 required'}), 400

    text1 = str(data['text1'])[:5000]
    text2 = str(data['text2'])[:5000]

    if len(text1.split()) < 5 or len(text2.split()) < 5:
        return jsonify({'similarity': 0.0})

    emb1 = get_embedding(text1).reshape(1, -1)
    emb2 = get_embedding(text2).reshape(1, -1)
    sim  = float(cosine_similarity(emb1, emb2)[0][0])
    sim  = max(0.0, min(1.0, sim))

    return jsonify({'similarity': round(sim, 4)})

@app.route('/compare_chunks', methods=['POST'])
def compare_chunks():
    """
    Compare one text against multiple corpus texts.
    Input:  { "text": "...", "corpus": ["...", "..."] }
    Output: { "best_similarity": 0.87, "scores": [0.87, 0.23, ...] }
    """
    data = request.get_json(force=True, silent=True)
    if not data or 'text' not in data or 'corpus' not in data:
        return jsonify({'error': 'text and corpus required'}), 400

    text   = str(data['text'])[:5000]
    corpus = [str(c)[:5000] for c in data['corpus'][:50]]  # max 50

    if not corpus:
        return jsonify({'best_similarity': 0.0, 'scores': []})

    emb_text   = get_embedding(text).reshape(1, -1)
    emb_corpus = model.encode(corpus, convert_to_numpy=True)
    scores     = cosine_similarity(emb_text, emb_corpus)[0].tolist()
    scores     = [max(0.0, min(1.0, float(s))) for s in scores]

    return jsonify({
        'best_similarity': round(max(scores), 4),
        'scores': [round(s, 4) for s in scores]
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
