# MealMind — Team Viva & Architecture Study Guide

> **Use this doc to prepare all 3 group members (`23L-0965`, `23L-0804`, `23L-0564`)** for the 20-minute evaluation (10 min presentation + 5 min live demo + 5 min Q&A).

---

## 1. What Problem Does MealMind Solve?

### Why Existing Recipe Apps Fail (e.g., AllRecipes, MyFitnessPal, Food.com)
Traditional recipe websites rely on **keyword search + manual dropdown filters**:
- **Mood / Craving failure:** Typing *"I want something fiery that will sizzle my tastebuds"* searches for the exact words `"sizzle"` and `"tastebuds"`. It misses *Peshawari Spicy Chicken Karahi* or *Szechuan Chili Beef* unless those exact words appear.
- **Natural Macro failure:** Typing *"calorie-dense meal for bulking over 850 calories"* cannot parse `"over 850"` or understand that `"bulking"` means high-calorie + high-protein.

### Why Plain Chatbots Fail (ChatGPT / LLM without RAG)
- **Hallucinated Macros:** A plain LLM guesses approximate calories and protein from memory instead of verified recipe data.
- **Not Grounded in Your Database:** In real-world nutrition/menu apps, the AI must recommend **only from your curated/verified recipes** (e.g., halal/macro-verified menu), not random internet recipes.
- **No Explainability:** You cannot see *which* source recipe backed the answer or *why* it was picked.

### How MealMind Solves Both
1. Understands natural cravings (**mood/vibe**) and **numeric macro goals** (*"at least 50g protein"*).
2. Retrieves **only** from our curated 100-recipe dataset using **Hybrid Search (BM25 + ChromaDB)**.
3. Grounds the LLM (`openai/gpt-oss-20b`) strictly in the top-3 retrieved recipes and displays retrieval scores live.

---

## 2. Tech Stack (No Models Trained From Scratch)

| Component | Tool / Model | Role |
| :--- | :--- | :--- |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face) | Pretrained 384-D Transformer that converts recipe text & queries into semantic vectors. |
| **Vector Database** | **ChromaDB** (`chromadb.PersistentClient`) | Stores all 100 recipe embeddings, documents, and metadata in a persistent HNSW cosine index (`data/chroma_db`). |
| **Lexical Retriever** | **BM25Okapi** (`rank-bm25`) | Classic IR keyword scorer (TF-IDF upgrade + document length normalization). |
| **Fusion & Intent** | **Reciprocal Rank Fusion (RRF, $k=60$)** + Regex Parser | Merges Dense + BM25 rankings and boosts recipes matching parsed macro/mood constraints. |
| **LLM Generator** | **Groq API** (`openai/gpt-oss-20b`) + Ollama fallback | Generates the final grounded response from the top-3 retrieved recipes. |
| **UI & Charts** | **Gradio 6.x** + **Plotly** + **Pandas** | 3-tab web interface (*Ask & Inspect Sources*, *RAG vs. Baseline*, *Evaluation & Recipes*). |

---

## 3. Dataset & Metadata Verbalization (`Macro Profile`)

### Dataset Files
- **`data/recipes.json`**: **100 curated recipes** across diverse cuisines (Pakistani, Indian, Middle Eastern, Chinese, Japanese, Korean, Thai, Italian, Mexican, American, Mediterranean).
- **`data/eval_qa.json`**: **45 held-out QA pairs** (9 per category across 5 categories): `vibe_flavor`, `nutrition_macro`, `recipe_detail`, `dietary_quick`, `paraphrase_semantic`.

### Why We Need Metadata Verbalization (`recipe_to_document()` in `rag_engine.py`)
Text retrievers (both BM25 and embeddings) **cannot do math** on raw JSON numbers like `"calories": 1020, "protein_g": 55` — they don't automatically know `1020` means *"bulking"* or `300` means *"cutting"*.

So before indexing, `recipe_to_document()` converts each JSON recipe into a clean prose document and auto-generates a **`Macro Profile`** line:
- `protein_g >= 40` → `"very high protein, protein-rich, muscle building"`
- `protein_g >= 25` → `"high protein, protein-rich"`
- `calories >= 700` → `"calorie-dense, high-calorie, bulking, mass gainer, energy-dense"`
- `calories <= 380` → `"low-calorie, light, cutting, weight loss friendly"`

### Exact Before → After Example
**Raw JSON (`data/recipes.json`):**
```json
{
  "id": "peanut_chocolate_bulking_smoothie",
  "name": "1000-Calorie Chocolate Peanut Butter Bulk Shake",
  "prep_time_min": 5, "cook_time_min": 0,
  "nutrition": {"calories": 1020, "protein_g": 55, "carbs_g": 108, "fat_g": 40},
  "flavor_profile": ["sweet", "chocolaty", "nutty", "creamy", "rich"],
  "dietary_tags": ["calorie-dense", "high-protein", "vegetarian", "quick", "bulking"]
}
```
**Verbalized Document (`recipe_to_document()`):**
```text
Recipe: 1000-Calorie Chocolate Peanut Butter Bulk Shake
Cuisine: American | Difficulty: Easy | Prep: 5 min | Cook: 0 min | Total Time: 5 min | Servings: 1
Nutrition per serving: 1020 kcal, 55g protein, 108g carbs, 40g fat
Macro Profile: very high protein, protein-rich, muscle building, calorie-dense, high-calorie, bulking, mass gainer, energy-dense
Flavor Profile: sweet, chocolaty, nutty, creamy, rich
Dietary Tags: calorie-dense, high-protein, vegetarian, quick, bulking, no-cook
Description: ...
Ingredients: ...
Steps: ...
```
This exact text is fed into **both**:
1. `all-MiniLM-L6-v2` → stored as a 384-D vector in **ChromaDB**
2. `tokenize()` → indexed by **`BM25Okapi`**

---

## 4. Dual Indexing & How R, A, and G Work (`rag_engine.py`)

### Where are R, A, and G?
- **R (Retrieval)** = `retrieve()`: **BM25**, **Dense (ChromaDB)**, and **Hybrid (RRF)** are the 3 ways to retrieve the **top-3** recipes.
- **A (Augmentation)** = `answer()`: Joins the top-3 recipe texts into `context` and injects them into the prompt (`"Retrieved Recipe Context:\n{context}\n\nUser Question: {query}"`).
- **G (Generation)** = `self.llm.generate()`: Sends the augmented prompt to **Groq (`openai/gpt-oss-20b`)**. *(Baseline `mode="none"` skips R and A and goes straight to G.)*

### How `BM25Okapi` Counts Across All Documents
We pass **all 100 tokenized documents as one list** at startup:
```python
self.bm25 = BM25Okapi([tokenize(doc) for doc in self.documents])
```
It builds a global Document Frequency table across all 100 recipes:
- Common words (`"chicken"`, `"cup"`) → **low IDF** (low weight).
- Rare words (`"cilbir"`, `"amchur"`, `"numbing"`) → **high IDF** (huge weight).

### Concrete Examples: Where Each Retriever Wins & Loses

1. **BM25 Wins, Dense Loses** (Rare Names / Exact Tags)
   - *Query:* `"What gives Turkish Cilbir its spicy kick?"`
     - **Dense (ChromaDB):** Doesn't know the rare name `"Cilbir"`; `"spicy kick"` pulls it to *Chicken Karahi* (#1) and *Chana Masala* (#2), pushing *Cilbir* to #3.
     - **BM25:** `"cilbir"` appears in only **1/100** recipes → puts *Turkish Cilbir* at **#1**.
   - *Query:* `"Give me a high-protein low-carb breakfast I can make fast"`
     - **Dense (ChromaDB):** `"high-protein"` + `"breakfast"` overpowered `"low-carb"`, returning high-carb oats/shakes (`62g–108g carbs`).
     - **BM25:** Matched exact tag `"low-carb"` → put *Egg & Cottage Cheese Scramble* (`11g carbs, 45g protein`) at **#1**.

2. **Dense Wins, BM25 Loses** (Synonyms / Paraphrases / Plural Quirks)
   - *Query:* `"I want a meal for weight gain with lots of energy"` (or *"calorie dense meal for bulking over 850 calories"*)
     - **Dense (ChromaDB):** Understands meaning and retrieves *1000-Calorie Bulk Shake*, *Loaded Steak Bulk Bowl*, *Mass-Gainer PB Oats*.
     - **BM25:** Fails on `"weight gain"` (if recipes say `"bulking"` instead), and on `"over 850 calories"`, the rare plural `"calories"` fooled BM25 into ranking *Creamy Tuscan Salmon Pasta* (`860 kcal`) #1 ahead of the `1020 kcal` Bulk Shake.

3. **Hybrid Wins Both** (RRF + NL Constraint Parser)
   $$\text{Hybrid Score} = 0.50 \cdot \frac{1}{60 + \text{rank}_{\text{dense}} + 1} + 0.50 \cdot \frac{1}{60 + \text{rank}_{\text{bm25}} + 1} + \text{Intent Bonus}$$
   - *Query:* `"I want a quick dairy-free high-protein toast ready in about 10 minutes"`
     - **Dense alone** returned sweet high-protein breakfasts; **Hybrid** combined BM25's exact `"toast"` + `"dairy-free"` match with `extract_query_constraints()` (`max_time=10`) to put *Tuna & Avocado Sourdough Toast* at **#1**.

### How Exact Numbers & Time Constraints Work (e.g., `"under 10 minutes"`)
1. **Regex matches by unit word** (`extract_query_constraints()`):
   - Time: `(under|in|within|less than) (\d+) (min|mins|minutes)` → sets `max_time = 10`
   - Protein: `... (\d+) (g|grams) (of) protein` → sets `min_protein = 50`
   - Calories: `... (\d+) (kcal|cal|calories)` → sets `min_calories` / `max_calories`
   Because each regex requires its unit word, time, protein, and calories never get confused.
2. **No reverse-conversion from embeddings:** You cannot convert a 384-D vector back into numbers. Instead, `self.recipes` (the original JSON) stays in memory linked by the same recipe `id` as ChromaDB, so Python checks `prep_time_min + cook_time_min <= max_time` directly on the JSON integers (both for Hybrid's `constraint_bonus` and for Gradio's hard sliders).
3. **Division of labor:**
   - **Embeddings (ChromaDB) + BM25** handle fuzzy cravings and meaning (*"sizzle my tastebuds"*, *"cozy comfort"*), which `if` statements cannot do.
   - **JSON integers** handle exact math (`<= 10 min`, `>= 50g protein`), which embeddings cannot do.

### 1-Paragraph Viva Summary (Say This!)
> *"First, we verbalize each JSON recipe by adding a `Macro Profile` text line. That text goes two places: into `all-MiniLM-L6-v2` to store 384-D vectors in **ChromaDB** (for semantic search), and into `BM25Okapi` as tokenized words (for exact keyword search — BM25 does not take vectors). At query time, both indexers rank all 100 recipes, **RRF + our regex constraint parser** fuses their ranks in Python to pick **one combined Top-3 list**, and we **augment** the prompt with those 3 recipes for the Groq LLM to generate a grounded answer."*

---

## 5. Evaluation Metrics & Results (`evaluate.py`)

### Retrieval Evaluation (100 Recipes, 45 QA Pairs)
- **Recall@1 / Recall@3:** Fraction of queries where the expected recipe appears in the top 1 (or top 3) results.
- **MRR (Mean Reciprocal Rank):** Average of $\frac{1}{\text{rank}}$ of the first correct recipe ($1.0$ if #1, $0.5$ if #2, $0.33$ if #3).

| Retriever | Recall@1 | Recall@3 | MRR | `paraphrase_semantic` R@3 |
| :--- | :---: | :---: | :---: | :---: |
| **BM25 (Lexical)** | **0.756** | **0.933** | **0.837** | **0.778** |
| **Dense (ChromaDB)** | 0.467 | 0.733 | 0.578 | 0.667 |
| **Hybrid (RRF + Intent)** | **0.756** | **0.889** | **0.815** | **0.778** |

> **Key Takeaway:** Scaling from 32 → 100 recipes adds semantic distractors (Dense Recall@3: `0.733`). **Hybrid RRF** ties BM25 on Recall@1 (`0.756`) and `paraphrase_semantic` (`0.778`), and lifts Dense-only Recall@3 to **`0.889` (+15.6 pts)**. Zero-shot MiniLM trailing BM25 on exact-keyword queries directly motivates Cross-Encoder reranking as future work.

### End-to-End Generation Evaluation (RAG vs. Baseline, 45 QA Pairs)
- **Token F1:** Word-level precision/recall overlap with the ground-truth reference answer.
- **ROUGE-L:** Longest Common Subsequence overlap (measures phrasing & completeness).
- **Semantic Similarity:** Cosine similarity between `all-MiniLM-L6-v2` embeddings of generated vs. reference answers.

| System (`openai/gpt-oss-20b`) | Token F1 | ROUGE-L | Sem. Similarity | Avg Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (No RAG)** | 0.1835 | 0.0756 | 0.4541 | 1.84s |
| **Dense RAG** | 0.2908 (+58.5%) | 0.1309 (+73.1%) | 0.6405 (+41.0%) | 3.15s |
| **Hybrid RAG (RRF)** | **0.3077** (+67.7%) | **0.1373** (+81.6%) | **0.6771** (+49.1%) | 3.17s |

---

## 6. Quick Viva Q&A Cheat-Sheet

- **Q: Why didn't you fine-tune a model?**
  - *A:* Our project falls under **Category 1: RAG Bot for Niche Domains**, not Category 2 (Fine-Tuning SLMs). The guidelines only require Base vs. Fine-tuned comparison for Category 2; our equivalent comparison is **Baseline (No-RAG) vs. Dense RAG vs. Hybrid RAG**.
- **Q: Why use Reciprocal Rank Fusion (RRF) instead of adding BM25 + Cosine scores?**
  - *A:* BM25 scores (`~4–7`) and cosine similarities (`0–1`) are on incompatible scales. RRF fuses their **ranks** cleanly using $\frac{1}{60 + \text{rank}}$.
- **Q: How do you handle prompt injection in RAG?**
  - *A:* Indirect prompt injection occurs when untrusted external documents contain hidden instructions. MealMind uses a **static, curated local JSON corpus** (no untrusted user uploads or web scraping) and has no tool/write access, minimizing attack surface.
- **Q: What are the limitations & future improvements?**
  - *A:* (1) Regex constraint parser handles common phrases but not complex negations (*"no nuts except almonds"*); (2) RAG adds ~1.6–2.2s latency; (3) Future work: Cross-Encoder reranking, PDF/recipe upload, and automatic allergen/serving-size scaling.
