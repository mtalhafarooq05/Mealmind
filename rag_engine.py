"""
MealMind Hybrid RAG Engine
Combines Dense Semantic Embeddings (all-MiniLM-L6-v2), Sparse Lexical Search (BM25),
Reciprocal Rank Fusion (RRF), and Natural-Language Nutrition/Mood Constraint Parsing.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
import requests
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "recipes.json"
DEFAULT_CHROMA_DIR = Path(__file__).parent / "data" / "chroma_db"


@dataclass
class RetrievedRecipe:
    recipe: Dict[str, Any]
    score: float
    dense_score: float
    bm25_score: float
    constraint_bonus: float
    document_text: str


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


def recipe_to_document(recipe: Dict[str, Any]) -> str:
    """Convert a structured recipe dict into a rich text chunk for embedding & BM25."""
    nut = recipe["nutrition"]
    total_time = recipe["prep_time_min"] + recipe["cook_time_min"]
    macro_labels = []
    if nut["protein_g"] >= 40:
        macro_labels.append("very high protein, protein-rich, muscle building")
    elif nut["protein_g"] >= 25:
        macro_labels.append("high protein, protein-rich")
    if nut["calories"] >= 700:
        macro_labels.append("calorie-dense, high-calorie, bulking, mass gainer, energy-dense")
    elif nut["calories"] <= 380:
        macro_labels.append("low-calorie, light, cutting, weight loss friendly")

    return (
        f"Recipe: {recipe['name']}\n"
        f"Cuisine: {recipe['cuisine']} | Difficulty: {recipe['difficulty']} | "
        f"Prep: {recipe['prep_time_min']} min | Cook: {recipe['cook_time_min']} min | "
        f"Total Time: {total_time} min | Servings: {recipe['servings']}\n"
        f"Nutrition per serving: {nut['calories']} kcal, {nut['protein_g']}g protein, "
        f"{nut['carbs_g']}g carbs, {nut['fat_g']}g fat\n"
        f"Macro Profile: {', '.join(macro_labels) if macro_labels else 'balanced macros'}\n"
        f"Flavor Profile: {', '.join(recipe['flavor_profile'])}\n"
        f"Dietary Tags: {', '.join(recipe['dietary_tags'])}\n"
        f"Description: {recipe['vibe_description']}\n"
        f"Ingredients: {'; '.join(recipe['ingredients'])}\n"
        f"Steps: {' '.join(f'{i+1}. {s}' for i, s in enumerate(recipe['steps']))}"
    )


def extract_query_constraints(query: str) -> Dict[str, Any]:
    """
    Lightweight NLP parser that extracts nutritional, dietary, time, and flavor
    intent signals from natural-language user queries.
    """
    q = query.lower()
    constraints: Dict[str, Any] = {
        "min_protein": None,
        "min_calories": None,
        "max_calories": None,
        "max_time": None,
        "dietary_tags": [],
        "want_spicy": False,
        "want_light": False,
        "want_creamy": False,
    }

    # Explicit numeric protein constraint: e.g., "over 40g protein", "at least 50 grams of protein"
    m_prot = re.search(r"(?:over|at least|>=|more than|\+)?\s*(\d{2,3})\s*(?:g|grams?)\s*(?:of\s+)?protein", q)
    if m_prot:
        constraints["min_protein"] = int(m_prot.group(1))
    elif any(kw in q for kw in ["protein rich", "protein-rich", "high protein", "high-protein", "muscle", "lots of protein"]):
        constraints["min_protein"] = 35

    # Explicit numeric calorie constraints
    m_cal_min = re.search(r"(?:over|at least|>=|more than|above)\s*(\d{3,4})\s*(?:kcal|cal|calories)", q)
    m_cal_max = re.search(r"(?:under|below|less than|<=|max)\s*(\d{3,4})\s*(?:kcal|cal|calories)", q)
    if m_cal_min:
        constraints["min_calories"] = int(m_cal_min.group(1))
    elif any(kw in q for kw in ["calorie dense", "calorie-dense", "high calorie", "high-calorie", "bulking", "bulk", "hardgainer", "mass gainer", "calorie surplus"]):
        constraints["min_calories"] = 700

    if m_cal_max:
        constraints["max_calories"] = int(m_cal_max.group(1))
    elif any(kw in q for kw in ["low calorie", "low-calorie", "cutting", "light meal", "weight loss"]):
        constraints["max_calories"] = 420

    # Explicit time constraint
    m_time = re.search(r"(?:under|in|within|less than)\s*(\d{1,2})\s*(?:min|mins|minutes)", q)
    if m_time:
        constraints["max_time"] = int(m_time.group(1))
    elif any(kw in q for kw in ["quick", "fast", "10-minute", "15-minute", "no-cook"]):
        constraints["max_time"] = 25

    # Dietary tags
    for tag in ["vegan", "vegetarian", "gluten-free", "dairy-free", "low-carb", "halal", "no-cook"]:
        if tag in q or tag.replace("-", " ") in q:
            constraints["dietary_tags"].append(tag)
    if "meatless" in q or "plant-based" in q or "plant based" in q:
        constraints["dietary_tags"].append("vegetarian")

    # Flavor / mood intent
    if any(kw in q for kw in ["sizzle", "fiery", "spicy", "heat", "hot", "chili", "tastebuds", "kick", "bold", "numbing"]):
        constraints["want_spicy"] = True
    if any(kw in q for kw in ["light", "fresh", "crisp", "refreshing", "clean"]):
        constraints["want_light"] = True
    if any(kw in q for kw in ["creamy", "cozy", "comfort", "velvety", "cheesy"]) or (
        "rich" in q and "protein rich" not in q and "protein-rich" not in q
    ):
        constraints["want_creamy"] = True

    return constraints


class LLMClient:
    """Groq API client (primary) with Ollama local fallback and extractive fallback."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "").strip()
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")

    @property
    def backend_name(self) -> str:
        if self.api_key:
            return f"Groq ({self.model})"
        return "Extractive Fallback (set GROQ_API_KEY in .env for LLM generation)"

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> Tuple[str, str]:
        """Return (answer_text, provider_used)."""
        if self.api_key:
            try:
                resp = requests.post(
                    self.groq_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": temperature,
                        "max_tokens": 550,
                    },
                    timeout=25,
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"].strip(), f"Groq ({self.model})"
            except requests.RequestException:
                pass

        # Try local Ollama if available
        try:
            resp = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": f"{system_prompt}\n\n{user_prompt}",
                    "stream": False,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip(), f"Ollama ({self.ollama_model})"
        except requests.RequestException:
            pass

        return "", "fallback"


class MealMindRAG:
    """
    Hybrid RAG system supporting:
      - 'bm25': lexical keyword retrieval
      - 'dense': semantic vector retrieval (all-MiniLM-L6-v2)
      - 'hybrid': Reciprocal Rank Fusion (Dense + BM25) + NL constraint bonus
      - 'none': Baseline LLM without retrieval
    """

    def __init__(
        self,
        data_path: Path | str = DEFAULT_DATA_PATH,
        chroma_dir: Path | str = DEFAULT_CHROMA_DIR,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        groq_model: Optional[str] = None,
    ):
        self.data_path = Path(data_path)
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.recipes: List[Dict[str, Any]] = json.load(f)

        self.documents: List[str] = [recipe_to_document(r) for r in self.recipes]
        self.id_to_idx: Dict[str, int] = {r["id"]: i for i, r in enumerate(self.recipes)}
        self.embedder = SentenceTransformer(embedding_model_name)

        # Persistent ChromaDB Vector Store (cosine similarity space)
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.chroma_client.get_or_create_collection(
            name="mealmind_recipes",
            metadata={"hnsw:space": "cosine"},
        )
        if self.collection.count() != len(self.recipes):
            # Rebuild index cleanly if dataset size changed
            try:
                self.chroma_client.delete_collection("mealmind_recipes")
            except Exception:
                pass
            self.collection = self.chroma_client.get_or_create_collection(
                name="mealmind_recipes",
                metadata={"hnsw:space": "cosine"},
            )
            embeddings = self.embedder.encode(
                self.documents, convert_to_numpy=True, normalize_embeddings=True
            ).tolist()
            self.collection.add(
                ids=[r["id"] for r in self.recipes],
                embeddings=embeddings,
                documents=self.documents,
                metadatas=[
                    {
                        "name": r["name"],
                        "cuisine": r["cuisine"],
                        "calories": int(r["nutrition"]["calories"]),
                        "protein_g": int(r["nutrition"]["protein_g"]),
                        "total_time_min": int(r["prep_time_min"] + r["cook_time_min"]),
                    }
                    for r in self.recipes
                ],
            )

        self.bm25 = BM25Okapi([tokenize(doc) for doc in self.documents])
        self.llm = LLMClient(model=groq_model)

    def _compute_constraint_bonus(self, recipe: Dict[str, Any], constraints: Dict[str, Any]) -> float:
        bonus = 0.0
        nut = recipe["nutrition"]
        total_time = recipe["prep_time_min"] + recipe["cook_time_min"]
        tags = set(recipe["dietary_tags"])
        flavors = set(recipe["flavor_profile"])

        if constraints["min_protein"] is not None:
            if nut["protein_g"] >= constraints["min_protein"]:
                bonus += 0.0015 + min((nut["protein_g"] - constraints["min_protein"]) * 0.00003, 0.0006)
        if constraints["min_calories"] is not None:
            if nut["calories"] >= constraints["min_calories"]:
                bonus += 0.0015 + min((nut["calories"] - constraints["min_calories"]) * 0.000003, 0.0006)
        if constraints["max_calories"] is not None:
            if nut["calories"] <= constraints["max_calories"]:
                bonus += 0.0015
        if constraints["max_time"] is not None:
            if total_time <= constraints["max_time"]:
                bonus += 0.0010
        for tag in constraints["dietary_tags"]:
            if tag in tags:
                bonus += 0.0012
        if constraints["want_spicy"] and ({"spicy", "fiery", "sizzling", "numbing"} & flavors):
            bonus += 0.0012
        if constraints["want_light"] and ({"light", "fresh", "crunchy"} & flavors):
            bonus += 0.0010
        if constraints["want_creamy"] and ({"creamy", "comforting", "rich", "cheesy"} & flavors):
            bonus += 0.0010

        return bonus

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        mode: str = "hybrid",
        dietary_filter: Optional[str] = None,
        min_protein: int = 0,
        max_calories: int = 1500,
        max_time: int = 120,
    ) -> List[RetrievedRecipe]:
        """Retrieve top_k recipes using 'bm25', 'dense', or 'hybrid' (RRF)."""
        if mode == "none":
            return []

        # Dense cosine similarity via ChromaDB vector store
        q_emb = self.embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True).tolist()
        chroma_res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=len(self.recipes),
            include=["distances"],
        )
        dense_scores = np.zeros(len(self.recipes), dtype=float)
        for rid, dist in zip(chroma_res["ids"][0], chroma_res["distances"][0]):
            if rid in self.id_to_idx:
                dense_scores[self.id_to_idx[rid]] = 1.0 - float(dist)

        # Sparse BM25 scores
        q_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens) if q_tokens else np.zeros(len(self.recipes))

        # Rank arrays (0 = best)
        dense_ranks = np.argsort(-dense_scores).argsort()
        bm25_ranks = np.argsort(-bm25_scores).argsort()

        constraints = extract_query_constraints(query)
        rrf_k = 60

        candidates: List[RetrievedRecipe] = []
        for idx, recipe in enumerate(self.recipes):
            nut = recipe["nutrition"]
            total_time = recipe["prep_time_min"] + recipe["cook_time_min"]

            # Hard UI filters
            if dietary_filter and dietary_filter != "All":
                if dietary_filter.lower() not in recipe["dietary_tags"]:
                    continue
            if nut["protein_g"] < min_protein:
                continue
            if nut["calories"] > max_calories:
                continue
            if total_time > max_time:
                continue

            d_score = float(dense_scores[idx])
            b_score = float(bm25_scores[idx])
            c_bonus = self._compute_constraint_bonus(recipe, constraints)

            if mode == "dense":
                final_score = d_score
            elif mode == "bm25":
                final_score = b_score
            else:  # hybrid (Reciprocal Rank Fusion + gentle NL constraint bonus)
                rrf_dense = 1.0 / (rrf_k + int(dense_ranks[idx]) + 1)
                rrf_bm25 = 1.0 / (rrf_k + int(bm25_ranks[idx]) + 1)
                final_score = 0.50 * rrf_dense + 0.50 * rrf_bm25 + c_bonus

            candidates.append(
                RetrievedRecipe(
                    recipe=recipe,
                    score=final_score,
                    dense_score=d_score,
                    bm25_score=b_score,
                    constraint_bonus=c_bonus,
                    document_text=self.documents[idx],
                )
            )

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]

    def _extractive_fallback(self, query: str, retrieved: List[RetrievedRecipe], mode: str) -> str:
        if mode == "none" or not retrieved:
            return (
                "No LLM API key is configured for Baseline (No-RAG) mode. "
                "Add `GROQ_API_KEY` to a `.env` file to compare live LLM generation."
            )

        lines = ["Based on the retrieved recipes in MealMind:"]
        for i, item in enumerate(retrieved, 1):
            r = item.recipe
            n = r["nutrition"]
            total_t = r["prep_time_min"] + r["cook_time_min"]
            lines.append(
                f"**{i}. {r['name']}** ({r['cuisine']} | {total_t} min | "
                f"{n['calories']} kcal, {n['protein_g']}g protein)\n"
                f"- *Vibe:* {r['vibe_description']}\n"
                f"- *Key Ingredients:* {', '.join(r['ingredients'][:5])}"
            )
        lines.append("*(Extractive summary shown because `GROQ_API_KEY` is not set or could not reach Groq.)*")
        return "\n\n".join(lines)

    def answer(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 3,
        dietary_filter: Optional[str] = None,
        min_protein: int = 0,
        max_calories: int = 1500,
        max_time: int = 120,
    ) -> Dict[str, Any]:
        """End-to-end query answering returning answer, retrieved sources, latency, and provider."""
        start = time.perf_counter()
        retrieved = self.retrieve(
            query=query,
            top_k=top_k,
            mode=mode,
            dietary_filter=dietary_filter,
            min_protein=min_protein,
            max_calories=max_calories,
            max_time=max_time,
        )

        if mode == "none":
            system_prompt = (
                "You are a helpful nutrition and cooking assistant. Answer the user's question "
                "concisely from general knowledge only. Do not invent exact citations."
            )
            user_prompt = f"User Question: {query}"
        else:
            context = "\n\n---\n\n".join(item.document_text for item in retrieved)
            system_prompt = (
                "You are MealMind, a grounded culinary and nutrition RAG assistant. "
                "Answer ONLY using the provided recipe context. Mention exact recipe names, "
                "calories, protein, and relevant ingredients/steps when helpful. "
                "If none of the retrieved recipes match a strict constraint, state that clearly."
            )
            user_prompt = f"Retrieved Recipe Context:\n{context}\n\nUser Question: {query}"

        answer_text, provider = self.llm.generate(system_prompt, user_prompt)
        if not answer_text:
            answer_text = self._extractive_fallback(query, retrieved, mode)

        latency = time.perf_counter() - start
        return {
            "query": query,
            "mode": mode,
            "answer": answer_text,
            "retrieved": retrieved,
            "latency_sec": round(latency, 4),
            "provider": provider,
            "parsed_constraints": extract_query_constraints(query),
        }
