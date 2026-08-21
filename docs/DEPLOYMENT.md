# VOX Production Deployment Guide (Phase 12)

This document provides complete, reproducible instructions for deploying the **VOX Evidence-Aware Multilingual Voice RAG** system using Docker and Docker Compose.

---

## Architecture Overview

```
                        [ Internet / Client / Judge ]
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │    VOX API Gateway & UI     │
                      │       (Port 8080:8080)      │
                      │  ─────────────────────────  │
                      │  • STT + LID + Analysis     │
                      │  • Grounding Engine         │
                      │  • LLM Generation           │
                      │  • Guardrails               │
                      │  • Tantivy BM25 Sparse      │
                      │  • Single-Page Demo UI      │
                      └──────────────┬──────────────┘
                                     │
                    [ Private Internal Network ]
                        (No host ports exposed)
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │    Qdrant Vector Database   │
                      │     (Private: 6333 only)    │
                      └─────────────────────────────┘
```

---

## Prerequisites

- **Docker** (v20.10+) and **Docker Compose** (v2.0+)
- Minimum System Requirements:
  - 2 CPU cores
  - 4 GB RAM
  - 10 GB disk space

---

## 1. Quickstart: 1-Command Deployment

From a clean repository checkout:

```bash
# 1. Start the stack (VOX API + Private Qdrant)
docker compose up -d --build

# 2. Check container health status
docker compose ps
```

Once running:
- **Demo UI**: Open `http://localhost:8080/` or `http://localhost:8080/demo` in any web browser.
- **Health Check**: `curl http://localhost:8080/health`
- **Prometheus Metrics**: `curl http://localhost:8080/metrics`

---

## 2. Environment Configuration

Copy `.env.example` to `.env` to configure external live APIs (optional):

```bash
cp .env.example .env
```

### Key Configuration Knobs

| Variable | Default | Purpose |
|---|---|---|
| `VOX_HOST` | `0.0.0.0` | API bind address |
| `VOX_PORT` | `8080` | API bind port |
| `RUST_LOG` | `info,vox=debug` | Tracing log level |
| `VOX_RETRIEVAL_MODE` | `oreo` | Embedded OREO engine with Tantivy BM25 + Qdrant |
| `VOX_OREO_VECTOR_STORE` | `qdrant` | Dense storage backend (`qdrant` or `memory`) |
| `VOX_OREO_QDRANT_URL` / `QDRANT_URL` | `http://qdrant:6333` | Internal private Qdrant URL |
| `VOX_STT_MODE` | `mock` | `mock` (deterministic) or `sarvam` (live STT) |
| `SARVAM_API_KEY` | *(empty)* | Sarvam AI API Key (required when `VOX_STT_MODE=sarvam`) |
| `VOX_LLM_MODE` | `extractive` | `extractive` (zero-latency) or `openai` (GPT-4o) |
| `LLM_API_KEY` | *(empty)* | OpenAI API Key (required when `VOX_LLM_MODE=openai`) |

> **Security Guarantee**: `qdrant` is placed on an isolated private internal network (`vox_internal: internal=true`) and does not publish port 6333 to the host machine.

---

## 3. Automated Deployment Verification

Run the verification suite to ensure all endpoints, grounding checks, and guardrails are operational:

**Linux / macOS:**
```bash
chmod +x scripts/verify_deployment.sh
./scripts/verify_deployment.sh http://localhost:8080
```

**Windows (PowerShell):**
```powershell
./scripts/verify_deployment.ps1 -BaseUrl "http://localhost:8080"
```

The verification suite checks:
1. `GET /health` (Status 200 OK, service=vox)
2. `GET /` (Demo UI HTML response)
3. `POST /v1/query` (English supported question → answer + evidence)
4. `POST /v1/query` (Tamil supported question → Tamil answer + evidence)
5. `POST /v1/query` (Unsupported question → grounded refusal `insufficient_context`)
6. `POST /v1/query` (Unsafe question → guardrail refusal `unsafe_input`, 0 ms retrieval cost)
7. `POST /v1/voice/query` (Voice base64 audio → STT transcript + evidence)

---

## 4. Operational Maintenance

### Viewing Live Logs
```bash
docker compose logs -f vox-api
```

### Stopping the Services
```bash
docker compose down
```

### Cleaning Data Volumes
```bash
docker compose down -v
```
