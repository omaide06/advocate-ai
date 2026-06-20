# ADVOCATE – The AI That's Designed to Disagree With You

> **A production-ready multi-agent AI backend that challenges your ideas with intellectual rigour.**

ADVOCATE analyses submitted ideas by scoring their quality, exposing hidden assumptions, generating the strongest possible counter-arguments, and delivering a final verdict — all through a modular, async Python backend that works with or without API keys.

---

## Architecture Overview

```
POST /analyze
      │
      ▼
┌─────────────────────────────────────────────────┐
│                  Orchestrator                   │
│                                                 │
│  ①  Quality Assessor                            │
│      → Score (1–5) + Attack Intensity           │
│                                                 │
│  ②  Assumption Scanner  ──┐  (concurrent)       │
│  ③  Steelman Generator  ──┘                     │
│                                                 │
│  ④  Formatter                                   │
│      → Verdict + Summary                        │
│                                                 │
│  ⑤  Persist → SQLite                            │
└─────────────────────────────────────────────────┘
      │
      ▼
  JSON Response
```

### Attack Intensity Mapping

| Score | Label     | Intensity  | What happens                                     |
|-------|-----------|------------|--------------------------------------------------|
| 1–2   | Very Poor / Poor | **aggressive** | 5–7 assumptions, 4–6 devastating counters |
| 3     | Moderate  | **balanced**   | 3–5 assumptions, 3–4 strong counters      |
| 4–5   | Strong / Excellent | **surgical** | 2–4 subtle assumptions, 2–3 precise counters |

### LLM Provider Resolution

Providers can be specified dynamically per-request, or resolved via environment variables:

1. Per-request `provider` and `api_key` → **Selected Provider** (Anthropic, OpenAI, Gemini, NVIDIA, or Mock)
2. `ANTHROPIC_API_KEY` set → **Claude** (Anthropic)
3. `OPENAI_API_KEY` set → **ChatGPT** (OpenAI)
4. `GEMINI_API_KEY` set → **Gemini** (Google)
5. No keys → **NVIDIA Free Hosted Models** (defaulting to `meta/llama-3.1-70b-instruct`)

---

## Project Structure

```
backend/
├── app/
│   ├── main.py                  # FastAPI app, lifespan, middleware
│   ├── api/
│   │   ├── analyze.py           # POST /analyze
│   │   ├── session.py           # GET /session/{id}, GET /sessions
│   │   └── health.py            # GET /health
│   ├── agents/
│   │   ├── orchestrator.py      # Coordinates all agents
│   │   ├── quality_assessor.py  # Scores idea 1–5, maps intensity
│   │   ├── assumption_scanner.py # Detects hidden assumptions
│   │   ├── steelman_generator.py # Generates strongest counter-args
│   │   └── formatter.py         # Synthesises verdict + summary
│   ├── services/
│   │   └── llm_service.py       # Anthropic / OpenAI / Mock abstraction
│   ├── database/
│   │   └── database.py          # Async SQLAlchemy engine + session
│   ├── models/
│   │   └── session.py           # AnalysisSession ORM model
│   ├── schemas/
│   │   ├── request.py           # AnalyzeRequest Pydantic schema
│   │   └── response.py          # AnalysisResponse, SessionListResponse, etc.
│   └── utils/
│       └── logger.py            # Structured colour logger
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python **3.11+**
- pip

### 1. Clone / navigate to the project

```bash
cd backend
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Want to use a real LLM?** Install the provider SDK:
> ```bash
> pip install anthropic    # for Claude
> pip install openai       # for GPT-4o
> ```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API key(s) if desired.
# The system works without any keys in mock mode.
```

### 5. Run the server

```bash
uvicorn app.main:app --reload
```

The server starts on **http://127.0.0.1:8000**.

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc
- **Health check**: http://127.0.0.1:8000/health

---

## API Reference

### `POST /analyze`

Run the full multi-agent analysis on an idea.

**Request body:**

```json
{
  "idea": "We should replace all public schools with privately-run charter schools.",
  "mode": "standard",
  "context": "Discussing US education policy reform.",
  "provider": "anthropic",
  "model": "claude-3-5-sonnet-20241022",
  "api_key": "sk-ant-..."
}
```

> **Note**: `provider`, `model`, and `api_key` are optional. If an `api_key` is provided in the request, it is used only for the duration of the request and is never stored. If omitted, the system falls back to environment variables or free NVIDIA models.

**Modes:**

| Mode       | Description                                                         |
|------------|---------------------------------------------------------------------|
| `standard` | Full pipeline: score → assumptions + counter-args (concurrent) → verdict |
| `quick`    | Score + assumptions only. No steelmanning, no formatter.           |
| `deep`     | Double steelman pass (exploiting assumptions), aggressive assumption scan. |

---

### `GET /models`

List all available LLM providers and their supported models. Useful for populating frontend dropdowns.

---

### `GET /health`

Liveness and readiness probe.

---

### `GET /session/{session_id}`

Retrieve a previously stored analysis by UUID.

---

### `GET /sessions`

List all sessions with optional pagination and mode filter.

**Query params:**

| Param    | Type    | Default | Description                        |
|----------|---------|---------|------------------------------------|
| `limit`  | integer | 20      | Max sessions to return (1–100).    |
| `offset` | integer | 0       | Sessions to skip (pagination).     |
| `mode`   | string  | –       | Filter by mode: standard/quick/deep |

---

## Example cURL Requests

### Analyse an idea (standard mode)

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "idea": "We should ban all social media platforms for users under 18 years old.",
    "mode": "standard",
    "context": "Debating youth mental health and digital policy."
  }'
```

### Quick mode (assumptions only)

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"idea": "Universal Basic Income will solve poverty.", "mode": "quick"}'
```

### Deep mode (maximum critique depth)

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"idea": "Fusion energy will be commercially viable by 2035.", "mode": "deep"}'
```

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### List sessions (paginated)

```bash
curl "http://127.0.0.1:8000/sessions?limit=10&offset=0&mode=standard"
```

### Get specific session

```bash
curl http://127.0.0.1:8000/session/YOUR-UUID-HERE
```

---

## Example JSON Response

```json
{
  "session_id": "3f8a1c2e-9b74-4d0e-bf35-12a4c7e891fd",
  "idea": "We should ban all social media platforms for users under 18 years old.",
  "mode": "standard",
  "score": 2.8,
  "quality_label": "Moderate",
  "attack_intensity": "balanced",
  "assumptions": [
    {
      "assumption": "Banning social media is technically enforceable at scale.",
      "severity": "critical",
      "explanation": "Age verification on the internet remains unsolved at scale. Every major attempt has been bypassed within weeks via VPNs, falsified birthdates, or parental account sharing."
    },
    {
      "assumption": "Social media is the primary cause of youth mental health decline.",
      "severity": "high",
      "explanation": "The causal link is contested in the scientific literature. Correlation between social media use and mental health issues does not establish causation; confounding variables (economic stress, academic pressure) are rarely controlled for."
    },
    {
      "assumption": "The harms of social media uniformly outweigh its benefits for all under-18s.",
      "severity": "high",
      "explanation": "Social media provides meaningful connection for isolated, LGBT+, or disabled youth who lack equivalent offline communities. A blanket ban removes these benefits without differentiation."
    }
  ],
  "counter_arguments": [
    {
      "argument": "Historical prohibition analogues — alcohol, cannabis — consistently demonstrate that blanket bans shift consumption underground rather than eliminating it, often creating more dangerous unregulated alternatives. A social media ban for minors would likely produce the same effect.",
      "strength": "devastating",
      "evidence_type": "historical"
    },
    {
      "argument": "The economic and political power of social media companies guarantees sustained legal challenges that would delay implementation by years, during which the harm continues unabated and legislative energy is consumed in litigation.",
      "strength": "strong",
      "evidence_type": "empirical"
    },
    {
      "argument": "Restricting access by age group requires identity verification infrastructure that creates new privacy harms — particularly for minors — that may exceed the harms the policy intends to prevent.",
      "strength": "strong",
      "evidence_type": "logical"
    }
  ],
  "verdict": "ADVOCATE finds the social media ban proposal to be moderately reasoned but structurally inadequate for the problem it targets. The idea correctly identifies a genuine public concern — youth mental health — but selects a blunt instrument that carries critical enforcement, privacy, and equity failures. The historical track record of prohibition-style interventions is uniformly poor; the technical barriers to age verification are currently insurmountable at meaningful scale; and the blanket nature of the proposal ignores the substantial benefits social media provides to marginalised youth. A more defensible policy would involve platform-level design standards (no infinite scroll, no algorithmic amplification of distressing content for minors) enforced through regulatory frameworks, rather than a ban. ADVOCATE recommends substantial redesign before this proposal could withstand serious policy scrutiny.",
  "summary": "The idea scores 2.8/5.0 — moderately coherent but critically flawed. Enforcement is technically infeasible at scale, the causal model is contested, and prohibition analogues predict failure. Platform design regulation is a stronger alternative.",
  "processing_time_seconds": 1.247,
  "created_at": "2026-06-17T11:30:00.000000",
  "llm_provider": "mock"
}
```

---

## Configuration Reference

| Variable            | Default                                    | Description                              |
|---------------------|--------------------------------------------|------------------------------------------|
| `ANTHROPIC_API_KEY` | –                                          | Anthropic Claude API key (optional).     |
| `ANTHROPIC_MODEL`   | `claude-3-5-haiku-20241022`                | Claude model to use.                     |
| `OPENAI_API_KEY`    | –                                          | OpenAI API key (optional).               |
| `OPENAI_MODEL`      | `gpt-4o-mini`                              | OpenAI model to use.                     |
| `GEMINI_API_KEY`    | –                                          | Google Gemini API key (optional).        |
| `GEMINI_MODEL`      | `gemini-2.0-flash`                         | Gemini model to use.                     |
| `NVIDIA_API_KEY`    | –                                          | NVIDIA API key for higher throughput (optional). |
| `DATABASE_URL`      | `sqlite+aiosqlite:///./advocate.db`        | SQLAlchemy async database URL.           |
| `LOG_LEVEL`         | `INFO`                                     | Logging verbosity (DEBUG/INFO/WARNING).  |
| `SQL_ECHO`          | `false`                                    | Echo SQL to stdout when `true`.          |
| `CORS_ORIGINS`      | `*`                                        | Allowed CORS origins (comma-separated).  |

---

## Running Without API Keys (Free NVIDIA Models)

The system detects the absence of API keys and automatically uses free hosted models provided by NVIDIA (defaulting to `meta/llama-3.1-70b-instruct`). **No configuration is required to run in this mode.**

```bash
# No .env needed – just run:
uvicorn app.main:app --reload
```

All endpoints function identically without keys. You can also explicitly request the built-in MockProvider by passing `"provider": "mock"` in your request body for deterministic, local testing.

---

## Development Notes

- **Async-first**: every database query and LLM call uses `await`. Never blocks the event loop.
- **Type hints everywhere**: all functions, methods, and dataclass fields are fully typed.
- **Graceful degradation**: JSON parse failures in any agent produce sensible fallback values rather than 500 errors.
- **Idempotent DB init**: `init_db()` uses `create_all` which is safe to call on every startup.
- **Concurrent agents**: Assumption Scanner and Steelman Generator run in parallel via `asyncio.gather` for ~2× throughput in standard mode.

---
