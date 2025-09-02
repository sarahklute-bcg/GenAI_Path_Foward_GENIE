#!/usr/bin/env python3
from __future__ import annotations
import os, json, re, time, argparse, pandas as pd
from pathlib import Path
from datetime import datetime
import ai_agentic_workflow as core

# -----------------------------
# Config
# -----------------------------

MISSING_FIELDS = ["impact_today_a","value_chain"]  # expand to your full set if desired
MAX_PASSES = 2
CONFIDENCE_FIELDS = ["metric_or_proof","outcome_or_impact"]

# -----------------------------
# Helpers
# -----------------------------

def empty(x):
    return x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and x.strip()=="")

def needs_enrichment(row: pd.Series) -> bool:
    return any(empty(row.get(f)) for f in MISSING_FIELDS)

def build_seed(row: pd.Series) -> str:
    cols = ["dem_industry_final","dem_sector_dialectica","function_standardized","function","workflow_text","company_name"]
    parts = [str(row[c]).strip() for c in cols if c in row and not empty(row[c])]
    seen=set(); uniq=[p for p in parts if not (p in seen or seen.add(p))]
    return " | ".join(uniq)

def critic_accepts(cases: list[dict]) -> bool:
    # accept if any case has non-empty metric & outcome
    for c in cases:
        if c.get("metric_or_proof","").strip() or c.get("outcome_or_impact","").strip():
            return True
    return False

def run_agentic(input_path: str, outfile: str, provider: str, model: str|None, api_version: str|None, topk: int, limit: int):
    df = pd.read_csv(input_path) if input_path.lower().endswith(".csv") else pd.read_excel(input_path)
    processed = 0
    aggregated = []

    for idx, row in df.iterrows():
        if limit and processed >= limit: break
        if not needs_enrichment(row):
            continue

        seed = build_seed(row)
        print(f"\n=== Row {idx} - Seed ===\n{seed}")
        accepted = False
        pass_num = 0
        gathered_cases = []

        while pass_num < MAX_PASSES and not accepted:
            pass_num += 1
            print(f"Pass {pass_num}/{MAX_PASSES}")

            # Use core.run_pipeline but do not write on each pass; collect and evaluate
            qb = core.refine_search(core.LLMClient(provider=provider, model=model, api_version=api_version), seed)
            docs = core.mock_search(qb, topk=topk)

            extracted = []
            llm = core.LLMClient(provider=provider, model=model, api_version=api_version)
            for i, d in enumerate(docs, start=1):
                data = core.extract_case(llm, d)
                data["rank"] = i
                extracted.append(data)

            print(f"  Extracted {len(extracted)} candidates")
            for e in extracted[:3]:
                print(f"   - {e.get('source_title','')}: {e.get('metric_or_proof','')} | {e.get('outcome_or_impact','')}")

            gathered_cases.extend(extracted)
            accepted = critic_accepts(extracted)
            if not accepted:
                print("  Critic: metrics/outcomes too weak; trying another pass...")
                time.sleep(0.5)

        # If still not accepted, keep whatever we have (could be empty)
        if not gathered_cases:
            continue

        # Persist (append)
        df_new = pd.DataFrame(gathered_cases)
        df_new["ingested_at"] = datetime.utcnow().isoformat(timespec="seconds")
        df_new["seed"] = seed
        df_new["row_index"] = idx
        for c in ["record","company_name","workflow_key","adoption_stage","dem_industry_final","dem_sector_dialectica","function_standardized","function","workflow_text","dem_region","dem_country"]:
            if c in row:
                df_new[c] = row[c]

        # Merge with existing Excel or create new
        if Path(outfile).exists():
            df_old = pd.read_excel(outfile, sheet_name="cases")
            df_all = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=["source_link","ai_use_case"], keep="first")
        else:
            df_all = df_new

        with pd.ExcelWriter(outfile, engine="xlsxwriter", mode="w") as writer:
            df_all.to_excel(writer, index=False, sheet_name="cases")

        processed += 1

    print(f"\nDone. Processed rows needing enrichment: {processed}. Output: {outfile}")

def main():
    ap = argparse.ArgumentParser(description="Agentic orchestrator over your Excel using ai_agentic_workflow tools")
    ap.add_argument("--input", required=True, help="CSV/XLSX of your data")
    ap.add_argument("--outfile", default="ai_value_cases.xlsx", help="Output Excel")
    ap.add_argument("--provider", choices=["auto","openai","azure"], default="auto")
    ap.add_argument("--model", default=None)
    ap.add_argument("--api_version", default=None)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    run_agentic(args.input, args.outfile, args.provider, args.model, args.api_version, args.topk, args.limit)

if __name__ == "__main__":
    main()
