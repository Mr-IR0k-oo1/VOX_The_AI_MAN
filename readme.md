# VOX – Multi-Model AI Assistant

VOX is a personal AI assistant that integrates multiple AI models, supports voice interaction, and can access the web for real-time information. It acts like a “Jarvis”-style assistant for critical thinking, coding tasks, and deep analysis.

---

## Features

* Supports multiple AI providers:

  * OpenAI GPT models
  * DeepSeek for deep analysis
  * Agentica DeepCoder for coding tasks
  * Grok (xAI) for specialized tasks
* Voice input (speech-to-text)
* Voice output (text-to-speech)
* Web browsing and retrieval
* Memory and context storage
* Modular and deployable via Docker

---

## Project Structure

```
vox/
 ├─ server.py           # FastAPI server
 ├─ model_adapters.py   # Multi-model API wrappers
 ├─ speech.py           # Voice recognition & TTS
 ├─ requirements.txt    # Python dependencies
 ├─ Dockerfile          # Docker build
 ├─ docker-compose.yml  # Docker Compose setup
 └─ .env                # Environment variables
```

---

## Requirements

* Docker and Docker Compose
* Python 3.10+ (for local testing)
* API keys for OpenAI, DeepSeek, HuggingFace/OpenRouter, Grok

---

## Environment Variables (`.env`)

```dotenv
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=ds_...
HF_API_TOKEN=hf_...
OPENROUTER_KEY=or_...
DEEPCODER_MODEL=agentica-org/deepcoder-14b-preview
GROK_API_KEY=grokkey...
SERVER_API_KEY=your-secret
```

---

## Installation & Deployment

### 1. Docker (Recommended)

```bash
# Build and start VOX
docker compose up --build -d

# View logs
docker compose logs -f
```

### 2. Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## Usage

### Chat via API

```bash
curl -X POST http://localhost:8000/vox/chat \
 -H "X-API-Key: your-secret" \
 -H "Content-Type: application/json" \
 -d '{"prompt":"Explain quantum computing simply","provider":"deepseek","voice":true}'
```

* `provider`: openai | deepseek | deepcoder | grok
* `voice`: true/false to enable text-to-speech reply

---

### Python Client Example

```python
import requests

API_KEY = "your-secret"
URL = "http://localhost:8000/vox/chat"
data = {"prompt":"Write a Python function to sort a list","provider":"deepcoder","voice":False}

resp = requests.post(URL, json=data, headers={"X-API-Key": API_KEY})
print(resp.json()["reply"])
```

---

## License

MIT License – free for personal and research use. Commercial use may require API provider licenses.

---

## Notes

* Ensure API keys are kept private.
* For production, run behind HTTPS with Nginx or Caddy.
* Respect websites’ Terms of Service when browsing.

---

I can also create a **more visual README with badges, a GIF demo, and Docker “one-liner” deploy instructions** to make it more appealing for GitHub.

Do you want me to do that next?
