# FastAPI Web Service Usage Guide

## 🚀 Quick Start

### Option 1: Run Directly

```bash
./run_api.sh
```

Then visit **http://localhost:8000/docs** in your browser!

### Option 2: Run with Python

```bash
source .venv/bin/activate
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option 3: Run with Docker

```bash
# Build image
docker build -t profanity-filter-api .

# Run container
docker run -p 8000:8000 profanity-filter-api
```

### Option 4: Docker Compose (Recommended for Production)

```bash
docker-compose up -d
```

---

## 📚 Interactive Documentation

**Swagger UI (Recommended for demos):**
- URL: http://localhost:8000/docs
- Click "Try it out" on any endpoint
- Enter your text
- Click "Execute"
- See results immediately!

**ReDoc (Alternative docs):**
- URL: http://localhost:8000/redoc
- Clean, searchable documentation

---

## 🔌 API Endpoints

### 1. **POST /predict** - Check Single Message

**Request:**
```json
{
  "text": "This is a test message",
  "model": "modernbert-multiclass",  // optional
  "mode": "auto"  // optional: "single", "hybrid", "auto"
}
```

**Response:**
```json
{
  "text": "This is a test message",
  "is_toxic": false,
  "confidence": 0.998,
  "toxicity_type": "Clean",
  "latency_ms": 15.5,
  "model_name": "modernbert-multiclass",
  "two_stage": true
}
```

**curl Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

---

### 2. **POST /batch** - Check Multiple Messages

**Request:**
```json
{
  "texts": [
    "Hello world",
    "Nice game!",
    "Good morning"
  ],
  "mode": "hybrid"
}
```

**Response:**
```json
{
  "results": [
    {
      "text": "Hello world",
      "is_toxic": false,
      "confidence": 0.99,
      ...
    },
    ...
  ],
  "total_count": 3,
  "toxic_count": 0,
  "clean_count": 3,
  "average_latency_ms": 12.3,
  "total_time_ms": 45.8
}
```

---

### 3. **GET /health** - Health Check

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "models_loaded": ["traditional-ml", "modernbert-multiclass"],
  "uptime_seconds": 123.45
}
```

---

### 4. **GET /models** - List Available Models

**Response:**
```json
[
  {
    "name": "modernbert-multiclass",
    "description": "ModernBERT Multi-Class (Level 4) - Best in-domain",
    "f1_gametox": 0.85,
    "latency_ms": "16",
    "best_for": "Gaming chat with toxicity types"
  },
  ...
]
```

---

## 💻 Client Examples

### Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "your message here"}
)
result = response.json()
print(f"Toxic: {result['is_toxic']}")
print(f"Type: {result['toxicity_type']}")
```

### JavaScript/TypeScript

```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: 'your message here'})
});

const result = await response.json();
console.log(`Toxic: ${result.is_toxic}`);
```

### curl

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "test message"}'

# Batch prediction
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["msg1", "msg2", "msg3"]}'

# Health check
curl http://localhost:8000/health
```

---

## 🎯 Demo Flow

1. **Start the server:**
   ```bash
   ./run_api.sh
   ```

2. **Open browser:**
   - Visit http://localhost:8000/docs

3. **Try the `/predict` endpoint:**
   - Click "Try it out"
   - Enter text: `"This is a test message"`
   - Click "Execute"
   - See results instantly!

4. **Try different models:**
   - Change `model` to `"modernbert-multiclass"`
   - See toxicity type breakdown

5. **Try batch processing:**
   - Go to `/batch` endpoint
   - Enter multiple messages
   - See summary statistics

---

## 🔧 Configuration

### Model Selection

**Auto/Hybrid Mode (Recommended):**
```json
{
  "text": "message",
  "mode": "auto"  // Uses Traditional ML + ModernBERT
}
```

**Specific Model:**
```json
{
  "text": "message",
  "model": "modernbert-multiclass",
  "mode": "single"
}
```

### Available Models

| Model | Use When | Latency |
|-------|----------|---------|
| `auto` (hybrid) | **Best overall** | 0.008-16ms |
| `modernbert-multiclass` | Gaming chat, need types | 16ms |
| `toxic-bert` | General use, cross-domain | 10ms |
| `traditional-ml` | High speed critical | 0.008ms |

---

## 📊 Performance

### Latency (from FastAPI)

- **First request:** ~5-6 seconds (model loading)
- **Subsequent requests:** 10-20ms
- **Hybrid mode (clean message):** ~0.01ms (fast-pass)
- **Batch processing:** ~12-15ms average per message

### Throughput

- **Single threaded:** ~60-100 requests/second
- **With workers:** Scale linearly

### Memory

- **Idle:** ~500MB
- **With models loaded:** ~2-3GB (ModernBERT + Traditional ML)

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build
docker build -t profanity-filter-api .

# Run
docker run -p 8000:8000 profanity-filter-api

# Visit http://localhost:8000/docs
```

### Docker Compose (Production)

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 🔒 Production Considerations

### Security

1. **Add API key authentication**
2. **Rate limiting** (use middleware)
3. **CORS configuration** (update in `main.py`)
4. **HTTPS** (use nginx reverse proxy)

### Scaling

1. **Multiple workers:**
   ```bash
   uvicorn main:app --workers 4
   ```

2. **Load balancer** (nginx, Traefik)

3. **Horizontal scaling** (Kubernetes, Docker Swarm)

### Monitoring

1. **Health checks:** `/health` endpoint
2. **Metrics:** Add Prometheus integration
3. **Logging:** Structured logging with JSON

---

## 🐛 Troubleshooting

### Issue: "Model not found"

**Solution:** Make sure models are trained:
```bash
cd level4-advanced
python train_modernbert_multiclass.py
```

### Issue: "Port 8000 already in use"

**Solution:** Change port:
```bash
uvicorn api.main:app --port 8001
```

### Issue: Slow first request

**Expected:** Model loading takes ~5-6 seconds on first request. This is normal.

---

## 📝 Notes

- **Auto mode** uses hybrid filtering (Traditional ML + ModernBERT)
- **Swagger UI** is perfect for demos - no Postman needed!
- **Interactive docs** update automatically as you modify the code
- **CORS** is enabled for all origins (configure for production)

---

**Ready to demo? Just run `./run_api.sh` and visit http://localhost:8000/docs!**
