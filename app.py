"""
Neoveli Semantic Similarity Microservice (PRO OPTIMIZED - FIXED)
Flask + sentence-transformers (RENDER SAFE)
"""

from flask import Flask, request, jsonify
import numpy as np
import os

# 🔥 IMPORTANT
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

MODEL_NAME = os.getenv('MODEL_NAME', 'paraphrase-MiniLM-L3-v2')

# 🔥 LAZY LOAD (CLAVE)
model = None

def get_model():
    global model
    if model is None:
        print(f"Loading model: {MODEL_NAME}...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME, device='cpu')
        print("Model loaded.")
    return model


def cosine_similarity_np(a, b):
    return float(np.dot(a, b.T))


def get_embedding(text: str):
    text = text[:1500]
    model = get_model()
    return model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': MODEL_NAME,
        'mode': 'lazy-load'
    })


@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json(force=True, silent=True)

    if not data or 'text1' not in data or 'text2' not in data:
        return jsonify({'error': 'text1 and text2 required'}), 400

    text1 = str(data['text1'])[:2000]
    text2 = str(data['text2'])[:2000]

    if len(text1.split()) < 5 or len(text2.split()) < 5:
        return jsonify({'similarity': 0.0})

    emb1 = get_embedding(text1).reshape(1, -1)
    emb2 = get_embedding(text2).reshape(1, -1)

    sim = cosine_similarity_np(emb1, emb2)
    sim = max(0.0, min(1.0, sim))

    return jsonify({'similarity': round(sim, 4)})


@app.route('/compare_chunks', methods=['POST'])
def compare_chunks():
    data = request.get_json(force=True, silent=True)

    if not data or 'text' not in data or 'corpus' not in data:
        return jsonify({'error': 'text and corpus required'}), 400

    text = str(data['text'])[:2000]
    corpus = [str(c)[:1500] for c in data['corpus'][:15]]

    if not corpus:
        return jsonify({'best_similarity': 0.0, 'scores': []})

    model = get_model()

    emb_text = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).reshape(1, -1)

    emb_corpus = model.encode(
        corpus,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = np.dot(emb_text, emb_corpus.T)[0].tolist()
    scores = [max(0.0, min(1.0, float(s))) for s in scores]

    return jsonify({
        'best_similarity': round(max(scores), 4),
        'scores': [round(s, 4) for s in scores]
    })


# 🔥 RENDER ENTRYPOINT
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
