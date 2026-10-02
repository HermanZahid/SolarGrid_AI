# SolarGrid AI — RAG Foundation

Pakistan-focused renewable-energy project intelligence platform.

## Current milestone
- official Pakistan source registry
- local PDF ingestion
- chunking
- zero-cost TF-IDF retrieval
- source/evidence display
- Streamlit workflow UI

## Free stack
Streamlit Community Cloud + GitHub + Groq free access + open-source Python packages.

## Setup
1. Download the official PDFs listed in `data/sources.json`.
2. Place them at their specified `local_file` paths.
3. `pip install -r requirements.txt`
4. `streamlit run app.py`

## Next
Local sentence-transformer embeddings → Groq generation → multi-agent orchestration → deterministic solar/financial engine → report generation.
