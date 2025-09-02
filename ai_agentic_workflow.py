#!/usr/bin/env python3
"""
AI Agentic Workflow Assistant (CLI/Notebook-friendly)
- Refines a short topic into rich search queries
- (Mock) searches a document source
- Uses GPT (OpenAI or Azure OpenAI) to extract structured "AI value cases"
- Appends results to an Excel database
- Optionally prints a top-3 summary

Usage (CLI):
  python ai_agentic_workflow.py "Retail" --topk 3 --outfile ai_value_cases.xlsx
  OPENAI_API_KEY=... python ai_agentic_workflow.py "AI in Healthcare" --provider openai --model gpt-4o-mini

Azure example:
  AZURE_OPENAI_API_KEY=... AZURE_OPENAI_ENDPOINT=... \
  python ai_agentic_workflow.py "Retail" --provider azure --model YOUR_DEPLOYMENT_NAME --api_version 2024-08-01-preview

Notes:
- If no API keys are found, runs in "offline" mode with a simple rule-based extractor to demonstrate the flow.
- Replace the MOCK_DOCS with your actual search results or plug in your search backend.
"""
from __future__ import annotations
import os, sys, json, argparse, re, textwrap
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
from datetime import datetime

# -----------------------------
# 1) Prompts (customize freely)
# -----------------------------

PROMPT_REFINE = """You are a research query optimizer for AI value-case discovery.
User seed topic: "{seed}"

Goal: produce a *concise* but *high-recall* set of search queries for finding documents that
describe real-world AI applications with measurable business impact.

Return STRICT JSON with keys:
- primary_query: one clear, specific query string that expands the seed
- boolean_query: a Lucene/Google-style query with OR/() and synonyms
- facets: list of 3-7 specific subtopics/angles (e.g., value chain steps, departments, data types)
- must_have_terms: list of terms that increase chance of documented, *measurable* impact (e.g., "ROI", "%", "AUC", "throughput", "baseline")
- exclude_terms: list of terms to downweight if found (e.g., "generic definition", "university course", "press release without metrics")

Constraints:
- Keep it industry-specific when possible (infer likely sector(s)).
- Prefer queries that surface *metrics*, *before/after*, or *controlled experiments*.
"""

PROMPT_EXTRACT = """You are an information extraction model. Read the SOURCE below and return one JSON object
with these keys exactly:
- industry
- sector
- ai_use_case
- workflow
- metric_or_proof
- outcome_or_impact
- source_title
- source_link

Rules:
- If a field is not present, write an empty string "" (do NOT hallucinate).
- 'workflow' should be HOW the AI is applied (steps, actors, where in process).
- 'metric_or_proof' should include concrete numbers or experimental evidence if present.
- 'outcome_or_impact' is the business result (revenue, cost, time, risk, quality, NPS, etc.).
- Keep answers short (<= 40 words per field). Avoid bullet points.
- Return STRICT JSON only, no commentary.

SOURCE TITLE: {title}
SOURCE LINK: {url}
SOURCE:
{content}
"""

# -----------------------------
# 2) LLM client abstraction
# -----------------------------

class LLMClient:
    """
    Unified thin wrapper for OpenAI or Azure OpenAI Chat Completions.
    Falls back to 'offline' mode if no credentials are found.
    """
    def __init__(self,
                 provider: str = "auto",
                 model: Optional[str] = None,
                 api_version: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.api_version = api_version

        self.mode = "offline"
        self._client = None

        # Try OpenAI first
        if provider in ("auto", "openai"):
            if os.getenv("OPENAI_API_KEY"):
                try:
                    from openai import OpenAI
                    self._client = OpenAI()  # uses env OPENAI_API_KEY and optional OPENAI_BASE_URL
                    self.mode = "openai"
                    if self.model is None:
                        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                except Exception:
                    pass

        # Try Azure next
        if self.mode == "offline" and provider in ("auto", "azure"):
            if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
                try:
                    from openai import AzureOpenAI
                    self._client = AzureOpenAI(
                        api_key=os.environ["AZURE_OPENAI_API_KEY"],
                        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                        api_version=self.api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
                    )
                    self.mode = "azure"
                    if self.model is None:
                        # In Azure, 'model' must be your *deployment name*
                        self.model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
                except Exception:
                    pass

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
        """
        Ask the model for JSON. If online, enforces JSON via strong instruction.
        If offline, returns a heuristic JSON.
        """
        if self.mode == "openai" or self.mode == "azure":
            # Use Chat Completions API for broad compatibility (OpenAI + Azure)
            # Ref: https://platform.openai.com/docs/guides/text-generation/chat-completions-api?lang=python
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},  # OpenAI supports; Azure often passes it through as well
                )
                text = resp.choices[0].message.content
                return json.loads(text)
            except Exception as e:
                # Fallback: try without response_format and parse
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt + "\nReturn ONLY valid JSON."},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                text = resp.choices[0].message.content
                try:
                    return json.loads(text)
                except Exception:
                    # last resort: extract json substring
                    m = re.search(r"\{.*\}", text, re.S)
                    return json.loads(m.group(0)) if m else {}
        else:
            # Offline heuristics (for demo/testing)
            return self._offline_json(system_prompt, user_prompt)

    def _offline_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Very lightweight heuristic outputs for demo runs without API keys."""
        if "query optimizer" in system_prompt:
            seed = re.search(r'User seed topic:\s*"(.*?)"', user_prompt, re.S)
            seed = seed.group(1) if seed else "AI"
            # naive expansions
            facets = ["value chain analytics", "demand forecasting", "pricing optimization", "customer service", "risk/fraud"]
            must = ["ROI", "%", "baseline", "A/B test", "throughput"]
            exclude = ["course", "tutorial", "generic definition", "press release", "no metrics"]
            boolean = f'("{seed}" OR "{seed} AI") AND ("case study" OR "benchmark" OR "before after" OR "ROI" OR "%")'
            return {
                "primary_query": f"{seed} AI case studies with metrics",
                "boolean_query": boolean,
                "facets": facets[:5],
                "must_have_terms": must,
                "exclude_terms": exclude,
            }
        else:
            # extraction heuristic: try to recover title/url and search for simple patterns
            title = re.search(r"SOURCE TITLE:\s*(.*)", user_prompt)
            url = re.search(r"SOURCE LINK:\s*(.*)", user_prompt)
            content_match = re.search(r"SOURCE:\n(.*)$", user_prompt, re.S)
            content = content_match.group(1).strip() if content_match else ""
            # Guess industry/sector
            industry = "Healthcare" if re.search(r"clinic|hospital|EHR|patient|ICU", content, re.I) else \
                       "Retail" if re.search(r"grocery|store|POS|SKU|inventory|fashion|apparel|footfall", content, re.I) else \
                       "Financial Services" if re.search(r"bank|credit|loan|fraud|trading", content, re.I) else ""
            sector = ""
            if industry == "Retail":
                sector = "Grocery" if "grocery" in content.lower() else ("Fashion/Apparel" if re.search(r"apparel|fashion", content, re.I) else "")
            if industry == "Healthcare":
                sector = "Provider" if re.search(r"hospital|clinic|EHR", content, re.I) else ("Payer" if re.search(r"claims|payer", content, re.I) else "")
            # Use-case extraction (very rough)
            uc = "Demand forecasting" if re.search(r"forecast|stockout|demand", content, re.I) else \
                 "Computer vision shelf analytics" if re.search(r"shelf|planogram|camera", content, re.I) else \
                 "Sepsis early warning" if re.search(r"sepsis|ICU|vitals|AUC", content, re.I) else \
                 "Triage/prior authorization automation" if re.search(r"prior authorization|triage|NLP", content, re.I) else ""
            # metric/proof: extract first percentage or AUC/ROC/F1
            metric = ""
            m = re.search(r"(\d{1,3}\.\d+%|\d{1,3}%|\bAUC\s*=\s*\d\.\d{2}|\bF1\s*=\s*\d\.\d{2})", content)
            if m:
                metric = m.group(0)
            # outcome: look for reductions/increases with units
            out = ""
            m2 = re.search(r"(reduc\w+|decreas\w+|increas\w+|improv\w+).{0,40}(\d{1,3}\.?\d?%|hours|days|bps|pp|million|\$[0-9,]+)", content, re.I)
            if m2:
                out = m2.group(0)
            return {
                "industry": industry or "",
                "sector": sector or "",
                "ai_use_case": uc or "",
                "workflow": "Extracted heuristically from text (offline demo).",
                "metric_or_proof": metric,
                "outcome_or_impact": out,
                "source_title": (title.group(1).strip() if title else ""),
                "source_link": (url.group(1).strip() if url else ""),
            }

# -----------------------------
# 3) Mock document source (replace with your real search)
# -----------------------------

MOCK_DOCS = [
    {
        "title": "Grocery Retail: Vision-based Shelf Monitoring",
        "url": "https://example.com/retail/shelf-vision",
        "content": """A large grocery chain deployed ceiling-mounted cameras and edge inferencing
        to detect shelf gaps and planogram non-compliance at aisle-level. The computer vision model
        triggered tasks to associates via mobile. A/B tests across 120 stores showed 18% fewer
        stockouts and +2.3% same-store sales on impacted categories. Average task-to-fill time
        dropped from 42 to 27 minutes (36% improvement).""",
        "meta": {"industry": "Retail", "sector": "Grocery", "date": "2024-05-01"}
    },
    {
        "title": "Fashion Retail: ML Demand Forecasting for New SKUs",
        "url": "https://example.com/retail/forecasting",
        "content": """An apparel retailer replaced heuristic forecasting with gradient-boosted trees
        and calendar/event features. Incorporating product embeddings improved MAPE by 14% vs. baseline.
        Allocation and markdown decisions were automated in weekly S&OP, yielding 9% lower end-of-season
        inventory and 4.1pp gross margin uplift.""",
        "meta": {"industry": "Retail", "sector": "Fashion/Apparel", "date": "2023-11-12"}
    },
    {
        "title": "Hospital ICU: Sepsis Early Warning with EHR Streaming",
        "url": "https://example.com/health/sepsis-ew",
        "content": """A health system integrated a gradient-boosted model over streaming vitals and labs.
        The model achieved AUC=0.89 in prospective validation. Alerts delivered to the care team reduced
        time-to-antibiotics by 1.8 hours on average and ICU mortality decreased by 11%. Governance included
        bias monitoring and weekly recalibration.""",
        "meta": {"industry": "Healthcare", "sector": "Provider", "date": "2024-02-08"}
    },
    {
        "title": "Payer Ops: NLP for Prior Authorization Triage",
        "url": "https://example.com/health/payer-prior-auth",
        "content": """A national payer used domain-tuned LLMs to triage prior auth requests.
        The workflow extracted key clinical entities, mapped rules, and routed ~38% of cases
        for straight-through decisions with human-in-the-loop exceptions. Denial overturn rate
        improved by 3.5pp and average handling time decreased by 22%. """,
        "meta": {"industry": "Healthcare", "sector": "Payer", "date": "2024-09-30"}
    },
    {
        "title": "Banking: Real-time Card Fraud using Graph ML",
        "url": "https://example.com/finserv/graph-fraud",
        "content": """A digital bank deployed graph neural networks over device, merchant, and cardholder
        relationships. Detection recall improved by 7.8pp at constant false-positive rate.
        Rollout with canary regions cut fraud losses by $18.2M annually while reducing manual
        reviews by 23%. """,
        "meta": {"industry": "Financial Services", "sector": "Banking", "date": "2023-09-05"}
    },
    {
        "title": "Omnichannel Retail: Personalization with Reinforcement Learning",
        "url": "https://example.com/retail/rl-reco",
        "content": """A specialty retailer implemented RL bandits for homepage modules and email subject lines.
        The system explored safely using Thompson sampling and guardrails. 90-day experiment lifted
        click-through by 12% and revenue per visitor by 5.6%. """,
        "meta": {"industry": "Retail", "sector": "Specialty", "date": "2024-07-15"}
    }
]

# -----------------------------
# 4) Simple keyword search (placeholder for your real search)
# -----------------------------

def refine_search(llm: LLMClient, seed: str) -> Dict[str, Any]:
    return llm.chat_json(
        system_prompt="You refine user topics into high-recall search queries.",
        user_prompt=PROMPT_REFINE.format(seed=seed),
        temperature=0.1,
    )

def score_doc(query_bundle: Dict[str, Any], doc: Dict[str, Any]) -> float:
    """Tiny lexical scorer: overlap of important terms."""
    text = (doc["title"] + " " + doc["content"]).lower()
    score = 0.0
    for term in query_bundle.get("must_have_terms", []):
        if term.lower() in text:
            score += 2.0
    # primary terms
    for t in re.findall(r"\w+", query_bundle.get("primary_query", "")):
        if t.lower() in text:
            score += 0.5
    # facets
    for f in query_bundle.get("facets", []):
        if f.lower() in text:
            score += 0.8
    return score

def mock_search(query_bundle: Dict[str, Any], topk: int = 5) -> List[Dict[str, Any]]:
    ranked = sorted(MOCK_DOCS, key=lambda d: score_doc(query_bundle, d), reverse=True)
    return ranked[:topk]

# -----------------------------
# 5) Extraction
# -----------------------------

def extract_case(llm: LLMClient, doc: Dict[str, Any]) -> Dict[str, Any]:
    prompt = PROMPT_EXTRACT.format(title=doc["title"], url=doc["url"], content=doc["content"])
    data = llm.chat_json(
        system_prompt="You extract structured AI value cases from documents.",
        user_prompt=prompt,
        temperature=0.0,
    )
    # Ensure all keys exist
    keys = ["industry","sector","ai_use_case","workflow","metric_or_proof","outcome_or_impact","source_title","source_link"]
    for k in keys:
        data.setdefault(k, "")
    return data

# -----------------------------
# 6) Persistence to Excel
# -----------------------------

DB_COLUMNS = ["industry","sector","ai_use_case","workflow","metric_or_proof","outcome_or_impact","source_title","source_link","ingested_at","seed","rank"]

def append_to_excel(cases: List[Dict[str, Any]], outfile: str, seed: str):
    df_new = pd.DataFrame(cases)
    df_new["ingested_at"] = datetime.utcnow().isoformat(timespec="seconds")
    df_new["seed"] = seed
    # if rank isn't present, create a simple rank order
    if "rank" not in df_new.columns:
        df_new["rank"] = range(1, len(df_new) + 1)
    # Reorder columns
    for col in DB_COLUMNS:
        if col not in df_new.columns:
            df_new[col] = ""
    df_new = df_new[DB_COLUMNS]

    if Path(outfile).exists():
        df_old = pd.read_excel(outfile)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        # simple dedupe by (source_link, ai_use_case)
        df_all = df_all.drop_duplicates(subset=["source_link","ai_use_case"], keep="first")
    else:
        df_all = df_new

    with pd.ExcelWriter(outfile, engine="xlsxwriter", mode="w") as writer:
        df_all.to_excel(writer, index=False, sheet_name="cases")

    return df_new, df_all

# -----------------------------
# 7) Orchestration
# -----------------------------

def run_pipeline(seed: str, topk: int, provider: str, model: Optional[str], outfile: str, api_version: Optional[str] = None, show_top: int = 3):
    llm = LLMClient(provider=provider, model=model, api_version=api_version)
    query_bundle = refine_search(llm, seed)
    results = mock_search(query_bundle, topk=topk)

    extracted = []
    for i, doc in enumerate(results, start=1):
        data = extract_case(llm, doc)
        data["rank"] = i
        extracted.append(data)

    df_new, df_all = append_to_excel(extracted, outfile, seed)

    # optional CLI summary
    topn = min(show_top, len(extracted))
    summary_rows = []
    for i in range(topn):
        r = extracted[i]
        summary_rows.append({
            "rank": r.get("rank", i+1),
            "industry": r["industry"],
            "use_case": r["ai_use_case"],
            "metric": r["metric_or_proof"],
            "impact": r["outcome_or_impact"],
            "source": r["source_title"],
        })
    return query_bundle, results, pd.DataFrame(summary_rows), df_new, df_all, llm.mode

def main():
    p = argparse.ArgumentParser(description="AI Agentic Workflow Assistant (mock search + extraction)")
    p.add_argument("seed", type=str, help='Seed topic, e.g., "Retail" or "AI in Healthcare"')
    p.add_argument("--topk", type=int, default=5, help="How many docs to process")
    p.add_argument("--provider", type=str, default="auto", choices=["auto","openai","azure"], help="LLM provider")
    p.add_argument("--model", type=str, default=None, help="Model name (OpenAI) or deployment name (Azure)")
    p.add_argument("--api_version", type=str, default=None, help="Azure OpenAI api_version (e.g., 2024-08-01-preview)")
    p.add_argument("--outfile", type=str, default="ai_value_cases.xlsx", help="Excel output path")
    p.add_argument("--show_top", type=int, default=3, help="Summary count to show")
    args = p.parse_args()

    qb, docs, df_summary, df_new, df_all, mode = run_pipeline(
        seed=args.seed,
        topk=args.topk,
        provider=args.provider,
        model=args.model,
        outfile=args.outfile,
        api_version=args.api_version,
        show_top=args.show_top,
    )

    print("\n=== Refined Query Bundle ===")
    print(json.dumps(qb, indent=2))
    print(f"\nProvider mode: {mode}")
    print("\n=== Top Results (mock) ===")
    for i, d in enumerate(docs, start=1):
        print(f"{i}. {d['title']}  |  {d['url']}")

    print("\n=== Top Cases (summary) ===")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()
