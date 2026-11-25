"""
MealMind — Gradio UI for Hybrid Culinary & Nutrition RAG
3 clean tabs:
  1. Ask & Inspect Sources (mood/vibe + macro queries with live retrieved chunks)
  2. RAG vs. Baseline Demo (side-by-side comparison)
  3. Evaluation Dashboard & Recipe Explorer
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from rag_engine import MealMindRAG, RetrievedRecipe

RESULTS_PATH = Path(__file__).parent / "results" / "eval_report.json"

rag = MealMindRAG()

DIETARY_OPTIONS = [
    "All",
    "high-protein",
    "calorie-dense",
    "vegetarian",
    "vegan",
    "gluten-free",
    "dairy-free",
    "low-carb",
    "quick",
    "halal",
    "bulking",
]

EXAMPLE_QUERIES = [
    "I want something super fiery that will sizzle my tastebuds",
    "I'm looking for a protein rich meal with at least 50g protein",
    "Give me a calorie dense meal for bulking over 850 calories",
    "Quick vegetarian high-protein dinner under 25 minutes",
    "Suggest a cozy, rich, creamy comfort dinner",
    "I'm cutting and need a light meal under 350 calories",
]


def format_sources(retrieved: List[RetrievedRecipe], mode: str, parsed_constraints: dict) -> str:
    if not retrieved:
        return "_No recipes matched the current filters. Try relaxing the sliders._"

    active_hints = []
    if parsed_constraints.get("min_protein"):
        active_hints.append(f"Protein >= {parsed_constraints['min_protein']}g")
    if parsed_constraints.get("min_calories"):
        active_hints.append(f"Calories >= {parsed_constraints['min_calories']}")
    if parsed_constraints.get("max_calories"):
        active_hints.append(f"Calories <= {parsed_constraints['max_calories']}")
    if parsed_constraints.get("max_time"):
        active_hints.append(f"Time <= {parsed_constraints['max_time']} min")
    if parsed_constraints.get("want_spicy"):
        active_hints.append("Flavor: Spicy/Fiery")
    if parsed_constraints.get("want_creamy"):
        active_hints.append("Flavor: Creamy/Comfort")
    if parsed_constraints.get("want_light"):
        active_hints.append("Flavor: Light/Fresh")

    lines = [f"### Retrieved Sources (`{mode.upper()}`)"]
    if active_hints:
        lines.append(f"**Detected Intent:** {', '.join(active_hints)}\n")

    for rank, item in enumerate(retrieved, 1):
        r = item.recipe
        n = r["nutrition"]
        total_t = r["prep_time_min"] + r["cook_time_min"]
        lines.append(
            f"**{rank}. {r['name']}** ({r['cuisine']} • {total_t} min)\n"
            f"- **Macros:** `{n['calories']} kcal` | `{n['protein_g']}g P` | `{n['carbs_g']}g C` | `{n['fat_g']}g F`\n"
            f"- **Flavor:** {', '.join(r['flavor_profile'])}\n"
            f"- **Scores:** Final `{item.score:.4f}` (Dense `{item.dense_score:.3f}`, BM25 `{item.bm25_score:.2f}`, Intent `+{item.constraint_bonus:.3f}`)\n"
        )
    return "\n".join(lines)


def handle_query(
    query: str,
    mode: str,
    dietary_filter: str,
    min_protein: int,
    max_calories: int,
    max_time: int,
) -> Tuple[str, str, str]:
    if not query.strip():
        return "Please enter a question or craving above.", "", ""
    res = rag.answer(
        query=query,
        mode=mode,
        top_k=3,
        dietary_filter=dietary_filter,
        min_protein=min_protein,
        max_calories=max_calories,
        max_time=max_time,
    )
    meta = f"Provider: **{res['provider']}** | Latency: **{res['latency_sec']:.2f}s**"
    sources_md = format_sources(res["retrieved"], mode, res["parsed_constraints"])
    return res["answer"], sources_md, meta


def compare_rag_vs_baseline(query: str) -> Tuple[str, str, str]:
    if not query.strip():
        return "Enter a query to compare.", "", ""
    rag_res = rag.answer(query=query, mode="hybrid", top_k=3)
    base_res = rag.answer(query=query, mode="none", top_k=0)
    rag_box = f"{rag_res['answer']}\n\n---\n*Latency: {rag_res['latency_sec']:.2f}s | Mode: Hybrid RAG*"
    base_box = f"{base_res['answer']}\n\n---\n*Latency: {base_res['latency_sec']:.2f}s | Mode: Baseline (No RAG)*"
    sources_md = format_sources(rag_res["retrieved"], "hybrid", rag_res["parsed_constraints"])
    return rag_box, base_box, sources_md


def load_eval_dashboard() -> Tuple[pd.DataFrame, go.Figure]:
    if not RESULTS_PATH.exists():
        empty_df = pd.DataFrame(
            [{"Note": "Run `python evaluate.py` first to populate evaluation metrics."}]
        )
        return empty_df, go.Figure()

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    ret = report.get("retrieval", {})
    rows = []
    for mode in ["bm25", "dense", "hybrid"]:
        if mode in ret:
            rows.append(
                {
                    "Retriever": mode.upper(),
                    "Recall@1": ret[mode]["recall@1"],
                    "Recall@3": ret[mode]["recall@3"],
                    "MRR": ret[mode]["mrr"],
                }
            )
    df = pd.DataFrame(rows)

    fig = go.Figure()
    for metric in ["Recall@1", "Recall@3", "MRR"]:
        fig.add_trace(go.Bar(name=metric, x=df["Retriever"], y=df[metric]))
    fig.update_layout(
        barmode="group",
        title=f"Retrieval Evaluation ({report.get('total_questions', 0)} QA Pairs)",
        yaxis=dict(range=[0, 1.05], title="Score"),
        height=340,
    )
    return df, fig


def build_recipe_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Recipe": r["name"],
                "Cuisine": r["cuisine"],
                "Time (min)": r["prep_time_min"] + r["cook_time_min"],
                "Calories": r["nutrition"]["calories"],
                "Protein (g)": r["nutrition"]["protein_g"],
                "Flavors": ", ".join(r["flavor_profile"][:4]),
                "Tags": ", ".join(r["dietary_tags"][:4]),
            }
            for r in rag.recipes
        ]
    )


with gr.Blocks(title="MealMind — Hybrid Culinary & Nutrition RAG") as demo:
    gr.Markdown(
        "# MealMind — Semantic Culinary & Nutrition RAG\n"
        "Ask by **craving/vibe** (*\"sizzle my tastebuds\"*), **macros** (*\"protein rich\"*, *\"calorie dense\"*), "
        "or **recipe details**. Powered by **Hybrid Retrieval (BM25 + `all-MiniLM-L6-v2` RRF)**."
    )

    with gr.Tab("Ask & Inspect Sources"):
        with gr.Row():
            with gr.Column(scale=3):
                query_box = gr.Textbox(
                    label="What are you in the mood for?",
                    placeholder="e.g., I want a fiery meal that will sizzle my tastebuds...",
                    lines=2,
                )
                with gr.Row():
                    mode_radio = gr.Radio(
                        choices=["hybrid", "dense", "bm25"],
                        value="hybrid",
                        label="Retrieval Mode",
                    )
                    diet_dd = gr.Dropdown(choices=DIETARY_OPTIONS, value="All", label="Dietary Filter")
                with gr.Accordion("Optional Macro & Time Filters", open=False):
                    with gr.Row():
                        min_prot = gr.Slider(0, 60, value=0, step=5, label="Min Protein (g)")
                        max_cal = gr.Slider(250, 1500, value=1500, step=50, label="Max Calories")
                        max_time = gr.Slider(10, 120, value=120, step=5, label="Max Total Time (min)")

                ask_btn = gr.Button("Ask MealMind", variant="primary")
                gr.Examples(examples=[[q] for q in EXAMPLE_QUERIES], inputs=[query_box])

                answer_out = gr.Markdown(label="MealMind Answer")
                meta_out = gr.Markdown()

            with gr.Column(scale=2):
                sources_out = gr.Markdown(value="### Retrieved Sources\n_Ask a question to inspect top-3 retrieved recipes and scores._")

        ask_btn.click(
            handle_query,
            inputs=[query_box, mode_radio, diet_dd, min_prot, max_cal, max_time],
            outputs=[answer_out, sources_out, meta_out],
        )
        query_box.submit(
            handle_query,
            inputs=[query_box, mode_radio, diet_dd, min_prot, max_cal, max_time],
            outputs=[answer_out, sources_out, meta_out],
        )

    with gr.Tab("RAG vs. Baseline"):
        gr.Markdown("Compare **Hybrid RAG (grounded in dataset)** side-by-side against **Baseline LLM (no retrieval)**.")
        cmp_input = gr.Textbox(
            label="Test Query",
            value="I want a calorie dense meal for bulking over 850 calories",
        )
        cmp_btn = gr.Button("Compare Side-by-Side", variant="primary")
        with gr.Row():
            rag_cmp_out = gr.Markdown(label="Hybrid RAG")
            base_cmp_out = gr.Markdown(label="Baseline (No RAG)")
        cmp_sources = gr.Markdown()

        cmp_btn.click(
            compare_rag_vs_baseline,
            inputs=[cmp_input],
            outputs=[rag_cmp_out, base_cmp_out, cmp_sources],
        )

    with gr.Tab("Evaluation & Recipes"):
        eval_df, eval_fig = load_eval_dashboard()
        gr.Markdown("### Retrieval Benchmark (BM25 vs. Dense vs. Hybrid RRF)")
        with gr.Row():
            gr.Dataframe(value=eval_df, interactive=False)
            gr.Plot(value=eval_fig)
        gr.Markdown(f"### Knowledge Base ({len(rag.recipes)} Curated Recipes)")
        gr.Dataframe(value=build_recipe_table(), interactive=False)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), share=True)
