# EDI Agent — Setup Guide

How to get the EDI agent running on a fresh environment.

## Quick Start (Docker Compose)

The fastest path. Requires only Docker with Compose v2.

### 1. Pick your GPU variant

```bash
# AMD GPU (ROCm)
docker compose up -d

# NVIDIA GPU
docker compose -f docker-compose.nvidia.yaml up -d

# CPU-only (use the NVIDIA compose file — Ollama falls back to CPU automatically)
docker compose -f docker-compose.nvidia.yaml up -d
```

### 2. Pull the LLM models

```bash
docker exec ollama ollama pull llama3.1:8b
docker exec ollama ollama pull llama3.2:3b
```

### 3. Verify

```bash
curl -s http://localhost:8008/v1/models | python3 -m json.tool
```

EDI is now running on port **8008** with OpenAI-compatible API. Open WebUI is at **http://localhost:3000**.

---

## Local Development Setup

For running EDI outside Docker (e.g., for development or debugging).

### Prerequisites

| Dependency | Version | Required | Notes |
|---|---|---|---|
| Python | 3.12+ | Yes | |
| PostgreSQL | 14+ with pgvector | Yes | Sessions, memories, heartbeats |
| Ollama | latest | Yes | Local LLM inference |
| Node.js | 20+ | Optional | Only for MCP stdio servers (searxng, filesystem) |
| OpenAI API key | — | Optional | For `text-embedding-3-small` embeddings (memory retrieval) |

### Step-by-step

**1. Clone and install**

```bash
git clone <repo-url> sr2 && cd sr2
uv sync --all-extras
```

**2. Start PostgreSQL with pgvector**

Option A — Docker (recommended):
```bash
docker run -d --name sr2-postgres \
  -e POSTGRES_USER=sr2 -e POSTGRES_PASSWORD=sr2 -e POSTGRES_DB=sr2 \
  -p 5432:5432 pgvector/pgvector:pg16
```

Option B — System PostgreSQL:
```bash
sudo apt install postgresql postgresql-16-pgvector   # Debian/Ubuntu
# Create the database:
sudo -u postgres createuser sr2 --pwprompt            # password: sr2
sudo -u postgres createdb sr2 --owner=sr2
# The pgvector extension is created automatically by SR2 on startup.
```

**3. Start Ollama and pull models**

```bash
# Install: https://ollama.com/download
ollama serve &                    # or: systemctl start ollama
ollama pull llama3.1:8b           # main model
ollama pull llama3.2:3b           # fast model (summarization, extraction)
```

**4. Configure environment**

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY if you want memory retrieval embeddings.
# Adjust DATABASE_URL / OLLAMA_API_BASE if your services aren't on defaults.
```

**5. Update agent config for local services**

The default `configs/agents/edi/agent.yaml` points to Docker service names (`postgres`, `ollama-exporter`). For local dev, override with environment variables or edit the config:

```yaml
# configs/agents/edi/agent.yaml — local overrides
runtime:
  database:
    url: "postgresql://sr2:sr2@localhost:5432/sr2"
  llm:
    model:
      api_base: http://localhost:11434    # direct Ollama, not exporter
    fast_model:
      api_base: http://localhost:11434
```

**6. Run**

```bash
sr2-agent configs/agents/edi --http --port 8008
```

**7. Test**

```bash
curl -X POST http://localhost:8008/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "hello"}'
```

---

## Services Reference

| Service | Default Port | Required | Purpose |
|---|---|---|---|
| PostgreSQL (pgvector) | 5432 | Yes | Session, memory, heartbeat storage |
| Ollama | 11434 | Yes | Local LLM inference |
| SR2 EDI (HTTP) | 8008 | Yes | Agent API |
| Open WebUI | 3000 | No | Chat UI frontend |
| SearXNG | 8765 | No | Web search MCP server |
| Galaxy Map | 8000 | No | Task management MCP server |
| Prometheus | 9090 | No | Metrics scraping |
| Grafana | 3080 | No | Dashboards |

## Docker Compose Files

| File | GPU | Notes |
|---|---|---|
| `docker-compose.yaml` | AMD (ROCm) | Default. Uses `ollama/ollama:rocm` with AMD device passthrough. |
| `docker-compose.nvidia.yaml` | NVIDIA / CPU | Uses `ollama/ollama:latest` with NVIDIA runtime. Works on CPU too. |
| `docker-compose.test.yaml` | — | PostgreSQL only, for running integration tests. |

## Database

All tables are **auto-created** on first startup. No manual migrations needed.

| Table | Purpose |
|---|---|
| `sessions` | Conversation turn history |
| `memories` | Extracted facts + pgvector embeddings |
| `heartbeats` | Scheduled future agent callbacks |

## MCP Servers (Optional)

EDI can connect to external tools via MCP. These are optional — EDI runs fine without them.

| Server | Transport | What it does | Needs |
|---|---|---|---|
| `searxng` | stdio (npx) | Web search + URL reading | Node.js 20+, SearXNG instance |
| `galaxy-map` | stdio | Task board management | `galaxy-map-mcp` binary |
| `filesystem` | stdio (npx) | File read/write/search | Node.js 20+ |

## Troubleshooting

**"Connection refused" to PostgreSQL**
- Ensure PostgreSQL is running and accessible on the configured host/port.
- Check that the `sr2` database exists: `psql -U sr2 -d sr2 -c '\dt'`

**Ollama models not loading**
- Verify models are pulled: `ollama list`
- Check Ollama logs: `journalctl -u ollama` or `docker logs ollama`

**Memory retrieval not working**
- Embeddings require `OPENAI_API_KEY` set in `.env` (uses `text-embedding-3-small`).
- Without it, memory extraction still works but semantic retrieval is disabled.

**MCP servers failing to connect**
- `npx`-based servers need Node.js 20+ in PATH.
- Check that the target service (SearXNG, Galaxy Map) is actually running.
