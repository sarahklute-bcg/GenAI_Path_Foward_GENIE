#!/usr/bin/env python3
import os, argparse, pandas as pd
from pathlib import Path
from datetime import datetime
import ai_agentic_workflow as core

DEFAULT_SEED_COLS = [
    "dem_industry_final",
    "dem_sector_dialectica",
    "function_standardized",
    "function",
    "workflow_text",
    "company_name",
]

DEFAULT_PROVENANCE_COLS = [
    "record","company_name","workflow_key","adoption_stage","change_aimed",
    "dem_industry_final","dem_sector_dialectica","function_standardized","function",
    "workflow_text","dem_region","dem_country","bffxai_score","bffxai_maturitystage",
    "bff_bucket","codes_present","impact_expected_p","impact_today_p","gap_p",
    "feasibility","peer_key","peer_adoption_rate","peer_impact_revenue",
    "peer_impact_prod_gains","peer_impact_headcount","peer_impact_cost_avoid",
    "peer_impact_risk_red","peer_impact_customer_exp","peer_impact_employee_exp",
    "p50_months","n_obs","peer_adoption_rate_n","peer_impact_revenue_n",
    "peer_impact_prod_gains_n","peer_impact_headcount_n","peer_impact_cost_avoid_n",
    "peer_impact_risk_red_n","peer_impact_customer_exp_n","peer_impact_employee_exp_n",
    "gap_p_n","feasibility_n","impact_kpi_mean_n","priority","tier","_sector_key",
    "_function_key","value_chain","match_flag"
]

def read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    elif path.lower().endswith((".xlsx",".xls")):
        return pd.read_excel(path)
    else:
        raise ValueError("Input must be .csv or .xlsx")

def build_seed(row: pd.Series, seed_cols):
    parts = []
    for c in seed_cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            parts.append(str(row[c]).strip())
    # de-dup while preserving order
    seen = set()
    uniq = [p for p in parts if not (p in seen or seen.add(p))]
    return " | ".join(uniq)

def main():
    ap = argparse.ArgumentParser(description="Batch launcher for ai_agentic_workflow.run_pipeline over tabular inputs")
    ap.add_argument("--input", required=True, help="Path to CSV/XLSX containing rows like the sample provided")
    ap.add_argument("--seed_cols", default=",".join(DEFAULT_SEED_COLS), help="Comma-separated column names to build seed text")
    ap.add_argument("--outfile", default="ai_value_cases.xlsx", help="Output Excel (cases sheet)")
    ap.add_argument("--cases_sheet", default="cases", help="Sheet name for extracted cases")
    ap.add_argument("--inputs_sheet", default="inputs", help="Sheet name to snapshot the input rows used")
    ap.add_argument("--topk", type=int, default=5, help="Top documents per seed to extract")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N rows (0 = all)")
    ap.add_argument("--provider", choices=["auto","openai","azure"], default="auto")
    ap.add_argument("--model", default=None, help="OpenAI model or Azure deployment name")
    ap.add_argument("--api_version", default=None, help="Azure OpenAI api version (e.g., 2024-08-01-preview)")
    ap.add_argument("--show_top", type=int, default=0, help="If >0, print summary per row")
    args = ap.parse_args()

    df = read_table(args.input)
    seed_cols = [c.strip() for c in args.seed_cols.split(",") if c.strip()]
    prov_cols = [c for c in DEFAULT_PROVENANCE_COLS if c in df.columns]

    # collect outputs
    all_cases = []
    used_rows = []

    n = len(df) if args.limit <= 0 else min(args.limit, len(df))
    for idx in range(n):
        row = df.iloc[idx]
        seed = build_seed(row, seed_cols)
        if not seed:
            continue

        qb, docs, df_summary, df_new, df_all, mode = core.run_pipeline(
            seed=seed,
            topk=args.topk,
            provider=args.provider,
            model=args.model,
            outfile=args.outfile,  # the function writes/updates this file, but we also aggregate below
            api_version=args.api_version,
            show_top=max(args.show_top, 0),
        )

        # augment with provenance
        augment = df_new.copy()
        for c in prov_cols:
            augment[c] = row.get(c, None)
        augment["source_seed"] = seed
        augment["row_index"] = idx
        all_cases.append(augment)
        used_rows.append(row)

        if args.show_top and not df_summary.empty:
            print(f"\n=== Row {idx} | Seed ===\n{seed}")
            print(df_summary.to_string(index=False))

    # Write a single clean workbook at the end (cases + inputs)
    if all_cases:
        cases_df = pd.concat(all_cases, ignore_index=True)
    else:
        cases_df = pd.DataFrame(columns=core.DB_COLUMNS + prov_cols + ["source_seed","row_index"])

    inputs_df = pd.DataFrame(used_rows) if used_rows else pd.DataFrame(columns=df.columns)

    with pd.ExcelWriter(args.outfile, engine="xlsxwriter", mode="w") as writer:
        cases_df.to_excel(writer, index=False, sheet_name=args.cases_sheet)
        inputs_df.to_excel(writer, index=False, sheet_name=args.inputs_sheet)

    print(f"\nWrote {len(cases_df)} extracted rows to: {args.outfile}")
    print(f"Provider mode: {mode}")

if __name__ == "__main__":
    main()
