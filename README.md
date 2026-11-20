# MealMind — Semantic Culinary & Nutrition RAG

A lean, evaluation-first **Hybrid RAG** application for culinary and nutrition question answering. Users can search naturally by **flavor/vibe** (*"something that will sizzle my tastebuds"*), **macros** (*"protein rich meal"*, *"calorie dense meal for bulking"*), or **recipe details**.

## NLP Methodology
1. **Dense Semantic Retrieval**: `sentence-transformers/all-MiniLM-L6-v2` embeddings with cosine similarity — handles vibe/mood queries where exact keywords don't appear in the recipe.
2. **Sparse Lexical Retrieval**: `BM25Okapi` — handles exact dish names, ingredients, and specific macro terms.
3. **Hybrid Fusion (RRF) + Lightweight NL Constraint Parsing**: Combines Dense + BM25 via Reciprocal Rank Fusion (`k=60`) and extracts soft nutritional/dietary/flavor constraints from natural language.
4. **Grounded Generation**: Groq LLM (`openai/gpt-oss-20b`, configurable via `GROQ_MODEL`) with Ollama local fallback and extractive fallback.

## Quick Setup

```powershell
cd "e:\NLP Project\mealmind"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with your free Groq API key (optional — retrieval & evaluation work without it):
```env
GROQ_API_KEY=your_groq_api_key_here
```

## Run Evaluation
```powershell
# Fast retrieval benchmark (BM25 vs Dense vs Hybrid across 36 QA pairs — no API key needed)
python evaluate.py

# Full end-to-end benchmark (Baseline No-RAG vs Dense RAG vs Hybrid RAG — uses GROQ_API_KEY)
python evaluate.py --with-generation
```

## Launch Gradio UI
```powershell
python app.py
```
