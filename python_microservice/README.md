# Neoveli Semantic Similarity Microservice

Flask + sentence-transformers (all-MiniLM-L6-v2) — free, local, no API keys needed.

## Requirements
- Python 3.8+
- ~500MB RAM
- A VPS (DigitalOcean, Linode, Render, Railway, etc.)

## Installation

```bash
cd python_microservice
pip install flask sentence-transformers torch scikit-learn
python app.py
```

## Production deployment (gunicorn)

```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5001 app:app
```

## Endpoints

### GET /health
Returns `{"status": "ok", "model": "all-MiniLM-L6-v2"}`

### POST /compare
```json
{ "text1": "...", "text2": "..." }
```
Returns: `{"similarity": 0.87}`

### POST /compare_chunks
```json
{ "text": "...", "corpus": ["...", "..."] }
```
Returns: `{"best_similarity": 0.87, "scores": [0.87, 0.23]}`

## Connecting to Neoveli

Once deployed, set in `public_html/config/config.php`:

```php
define('SEMANTIC_SERVICE_URL', 'http://YOUR_VPS_IP:5001');
```

The PHP engine will automatically use semantic similarity when this URL is set,
and fall back to TF-IDF cosine similarity when it's not available.

## Security

Add a firewall rule to only allow your web server's IP to access port 5001.
Or use nginx as a reverse proxy with basic auth.
