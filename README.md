# ASBOS — Autonomous Sneaker Brand Operating System

<p align="center">
  <strong>Arétier (Ah-ray-tee-ay)</strong><br>
  <em>"The Maker of Excellence"</em><br>
  <sub>Arete (ἀρετή) + -ier · Valour · Strength · Persistence</sub>
</p>

---

A **stateful multi-agent LangGraph system** that autonomously operates a virtual sneaker brand across the full product lifecycle:

```
Trend Research → Design + Pricing + Drop Timing (parallel) → Copywriting → Adversarial Critique → Launch
```

## Architecture

### Agents
| Agent | Responsibility | Key Tools |
|---|---|---|
| **Trend Research** | Scrape Hypebeast + Reddit, extract aesthetic keywords | feedparser, PRAW, ChromaDB RAG |
| **Design** | Write design brief, generate image, score with CLIP | Gemini Imagen, CLIP (local) |
| **Pricing** | Set retail price using resale comps | ChromaDB market_data, StockX dataset |
| **Drop Timer** | Pick optimal drop date/time | ChromaDB drop_timing, historical CSV |
| **Copywriter** | Write product description + tweet drafts | Gemini, brand_voice RAG |
| **Critic** | Adversarial evaluation, revision requests | Deterministic rubric + LLM |

### Graph Topology
- Design, Pricing, and Drop Timer agents run **in parallel** (LangGraph fan-out)
- Critic gates every launch — approval score ≥ 0.75 required
- Max 3 revision iterations before auto-fail
- Sycophancy prevention: Critic must find at least one issue

### Tech Stack (100% Free)
- **LLM + Image Gen**: Google Gemini 1.5 Flash + Imagen 3
- **Vector DB**: ChromaDB (local, disk-persisted)
- **Embeddings**: all-MiniLM-L6-v2 (runs on your GPU)
- **Image Evaluation**: CLIP ViT-B/32 (local, ~300MB VRAM)
- **Orchestration**: LangGraph
- **Frontend**: Streamlit
- **Observability**: LangSmith

---

## Setup

### 1. Install dependencies
```bash
cd asbos
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Build the vector store (one-time)
```bash
python ingestion/build_vectorstore.py
```

### 4. Run the app
```bash
streamlit run app/streamlit_app.py
```

---

## Getting API Keys (All Free)

| Service | URL | Free Tier |
|---|---|---|
| Gemini API | [aistudio.google.com](https://aistudio.google.com) | 15 RPM, 1M TPM/day |
| LangSmith | [smith.langchain.com](https://smith.langchain.com) | Free observability |
| Reddit API | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) | Free read access |

---

## Project Structure
```
asbos/
├── agents/          # 6 specialist agents
├── graph/           # LangGraph state machine
├── tools/           # RAG, image gen, CLIP, web scraping, pricing
├── evaluation/      # Deterministic critic rubric
├── ingestion/       # One-time vector store build script
├── knowledge_base/  # brand_voice.md + drop_history.jsonl
├── data/            # historical_drops.csv
├── app/             # Streamlit frontend
└── requirements.txt
```

---

## Evaluation Metrics
- **Design**: CLIP cosine similarity ≥ 0.28 (text-image alignment)
- **Pricing**: Retail price in [$80, $500], resale premium ≥ 15%
- **Timing**: Drop ≥ 7 days out, preferred Thu/Fri/Sat
- **Copy**: 100–200 words, no banned openers, no price mentions
- **Composite**: Weighted average (Design 30%, Pricing 25%, Timing 20%, Copy 25%)

---

*Built for Agentic AI / LLM product roles — demonstrating multi-agent orchestration, self-evaluation, and production-grade state management.*
