#!/usr/bin/env python3
"""
Agentic Enricher for Existing AI Use-Case Excel
- Reads your Excel
- Detects rows with missing/weak fields
- Plans prompts per row (planner)
- Calls LLM to fill fields (worker)
- Validates + scores JSON (critic)
- Retries with feedback if needed
- Appends enriched rows to a new Excel

Usage (OpenAI):
  OPENAI_API_KEY=... python agentic_enricher.py --input ai_use_cases.xlsx --sheet Sheet1 --outfile enriched_ai_use_cases.xlsx --provider openai --model gpt-4o-mini

Windows PowerShell:
  $env:OPENAI_API_KEY="sk-..."
  python agentic_enricher.py --input ai_use_cases.xlsx --sheet Sheet1 --outfile enriched_ai_use_cases.xlsx --provider openai --model gpt-4o-mini

Azure:
  AZURE_OPENAI_API_KEY=... AZURE_OPENAI_ENDPOINT=... python agentic_enricher.py --input ... --provider azure --model YOUR_DEPLOYMENT --api_version 2024-08-01-preview
"""
from __future__ import annotations
import os, sys, argparse, json, re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
from datetime import datetime

# Optional: pyyaml for external config (shipped alongside)
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# --------------- LLM Client (OpenAI / Azure) ---------------

class LLMClient:
    def __init__(self, provider="openai", model=None, api_version=None):
        self.provider = provider
        self.model = model
        self.api_version = api_version
        self.mode = "offline"
        self._client = None

        if provider in ("openai","auto"):
            if os.getenv("OPENAI_API_KEY"):
                try:
                    from openai import OpenAI
                    self._client = OpenAI()
                    self.mode = "openai"
                    if not self.model:
                        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                except Exception:
                    pass
        if self.mode == "offline" and provider in ("azure","auto"):
            if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
                try:
                    from openai import AzureOpenAI
                    self._client = AzureOpenAI(
                        api_key=os.environ["AZURE_OPENAI_API_KEY"],
                        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                        api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION","2024-08-01-preview"),
                    )
                    self.mode = "azure"
                    if not self.model:
                        self.model = os.getenv("AZURE_OPENAI_DEPLOYMENT","gpt-4o-mini")
                except Exception:
                    pass

    def chat_json(self, system_prompt: str, user_prompt: str, temperature=0.1) -> Dict[str,Any]:
        if self.mode in ("openai","azure"):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    messages=[
                        {"role":"system","content":system_prompt},
                        {"role":"user","content":user_prompt}
                    ],
                    response_format={"type":"json_object"}
                )
                text = resp.choices[0].message.content
                return json.loads(text)
            except Exception:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    messages=[
                        {"role":"system","content":system_prompt+"\nReturn ONLY valid JSON."},
                        {"role":"user","content":user_prompt}
                    ]
                )
                text = resp.choices[0].message.content
                try:
                    return json.loads(text)
                except Exception:
                    m = re.search(r"\{.*\}", text, re.S)
                    return json.loads(m.group(0)) if m else {}
        # offline fallback
        return {}

# --------------- Agentic Prompts ---------------

PLANNER_PROMPT = """You are a planning agent that prepares gap-filling instructions for an AI table.
Given a row (company, industry, sector, function, workflow, region), select which fields to fill
from the target list and craft precise guidance and constraints.

Return STRICT JSON with:
- target_fields: array of keys to fill (subset of allowed_fields)
- guidance: short text with hints based on context (<= 40 words)
- musts: array of short constraints (e.g., "no hallucinated metrics", "≤ 25 words per field")

allowed_fields: {allowed_fields}
row_context:
{row_json}
"""

WORKER_PROMPT = """You are a BCG analyst enriching an AI use-case row.
Return STRICT JSON with EXACT keys from target_fields; if unknown, use "" (empty string).

Constraints:
- Do NOT invent facts or metrics. Prefer generic but plausible phrasing only if implied.
- ≤ 35 words per field. No lists or bullets.
- US English, concise, business tone.

target_fields: {target_fields}
guidance: {guidance}
musts: {musts}

Row context:
{row_json}
"""

CRITIC_PROMPT = """You are a critic validating a JSON object for schema and quality.
Check: keys subset of allowed_fields, string values <= 200 chars, not all empty; avoid hallucinated metrics.
Return STRICT JSON:
- ok: true/false
- issues: array of short strings
- fix_suggestion: one sentence describing how to fix

allowed_fields: {allowed_fields}
proposed_json:
{proposed_json}
"""

# --------------- Config Defaults ---------------

DEFAULT_ALLOWED_FIELDS = [
    "impact_today_a","impact_expected_c","value_chain",
    "months_to_fully_deployed","priority","tier"
]

# --------------- I/O Helpers ---------------

def read_table(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet or 0)

def write_outputs(df_inputs: pd.DataFrame, df_enriched: pd.DataFrame, outfile: str):
    with pd.ExcelWriter(outfile, engine="xlsxwriter", mode="w") as w:
        df_inputs.to_excel(w, index=False, sheet_name="inputs")
        df_enriched.to_excel(w, index=False, sheet_name="enriched")

# --------------- Agent Orchestrator ---------------

def is_empty(v) -> bool:
    if v is None: return True
    if isinstance(v, float) and pd.isna(v): return True
    return (str(v).strip() == "")

def detect_gaps(row: pd.Series, target_fields: List[str]) -> List[str]:
    missing = []
    for k in target_fields:
        if k in row and not is_empty(row[k]):
            continue
        missing.append(k)
    return missing

def run_enricher(input_path: str, sheet: Optional[str], outfile: str,
                 provider: str, model: Optional[str], api_version: Optional[str],
                 batch_size: int, max_rows: int, show_progress: bool,
                 allowed_fields: List[str]) -> pd.DataFrame:

    llm = LLMClient(provider=provider, model=model, api_version=api_version)
    df = read_table(input_path, sheet)
    records = []
    n = len(df) if max_rows <= 0 else min(max_rows, len(df))

    for i in range(n):
        row = df.iloc[i]
        # 1) Planner decides which targets to fill (based on what's empty)
        gaps = detect_gaps(row, allowed_fields)
        if not gaps:
            continue

        planner = llm.chat_json(
            system_prompt="You plan gap-filling tasks for AI use-case rows.",
            user_prompt=PLANNER_PROMPT.format(
                allowed_fields=json.dumps(allowed_fields),
                row_json=json.dumps(row.to_dict(), ensure_ascii=False, default=str)
            ),
            temperature=0.0
        )
        target_fields = [k for k in planner.get("target_fields", gaps) if k in allowed_fields]

        # 2) Worker produces candidate JSON
        worker = llm.chat_json(
            system_prompt="You fill missing fields for an AI table as strict JSON.",
            user_prompt=WORKER_PROMPT.format(
                target_fields=json.dumps(target_fields),
                guidance=json.dumps(planner.get("guidance","")),
                musts=json.dumps(planner.get("musts", [])),
                row_json=json.dumps(row.to_dict(), ensure_ascii=False, default=str)
            ),
            temperature=0.1
        )

        # 3) Critic validates
        critic = llm.chat_json(
            system_prompt="You validate JSON outputs for schema and quality.",
            user_prompt=CRITIC_PROMPT.format(
                allowed_fields=json.dumps(allowed_fields),
                proposed_json=json.dumps(worker, ensure_ascii=False, default=str)
            ),
            temperature=0.0
        )

        # 4) Simple retry if critic flags issues
        if not critic.get("ok", True):
            fix_note = "; ".join(critic.get("issues", []))[:200]
            worker = llm.chat_json(
                system_prompt="You fix JSON outputs based on critic feedback. Return strict JSON with same keys.",
                user_prompt=(
                    WORKER_PROMPT.format(
                        target_fields=json.dumps(target_fields),
                        guidance=json.dumps(planner.get("guidance","")),
                        musts=json.dumps(planner.get("musts", [])),
                        row_json=json.dumps(row.to_dict(), ensure_ascii=False, default=str)
                    )
                    + f"\n\nCRITIC_FEEDBACK: {fix_note}"
                ),
                temperature=0.0
            )

        # 5) Merge back into the row
        merged = row.to_dict()
        for k in allowed_fields:
            if k in worker:
                merged[k] = worker.get(k, merged.get(k, ""))

        merged["__enriched_at"] = datetime.utcnow().isoformat(timespec="seconds")
        merged["__row_index"] = i
        records.append(merged)

        if show_progress and (i+1) % max(1,batch_size) == 0:
            print(f"Processed {i+1}/{n} rows ...")

    # Output dataframe
    df_out = pd.DataFrame(records) if records else pd.DataFrame(columns=list(df.columns)+["__enriched_at","__row_index"])
    write_outputs(df.iloc[:n], df_out, outfile)
    return df_out

def main():
    ap = argparse.ArgumentParser(description="Agentic Enricher for Existing Excel AI Use-Cases")
    ap.add_argument("--input", required=True, help="Path to CSV/XLSX")
    ap.add_argument("--sheet", default=None, help="Excel sheet name (default first sheet)")
    ap.add_argument("--outfile", default="enriched_ai_use_cases.xlsx")
    ap.add_argument("--provider", choices=["openai","azure","auto"], default="openai")
    ap.add_argument("--model", default=None)
    ap.add_argument("--api_version", default=None)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--max_rows", type=int, default=0, help="0 = all")
    ap.add_argument("--show_progress", action="store_true")
    ap.add_argument("--config", default=None, help="YAML config with allowed_fields etc.")
    args = ap.parse_args()

    allowed_fields = DEFAULT_ALLOWED_FIELDS
    if args.config and Path(args.config).exists():
        try:
            data = yaml.safe_load(Path(args.config).read_text())
            allowed_fields = data.get("allowed_fields", allowed_fields)
        except Exception:
            pass

    df_out = run_enricher(
        input_path=args.input,
        sheet=args.sheet,
        outfile=args.outfile,
        provider=args.provider,
        model=args.model,
        api_version=args.api_version,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        show_progress=args.show_progress,
        allowed_fields=allowed_fields,
    )
    print(f"Enriched {len(df_out)} rows → {args.outfile}")

if __name__ == "__main__":
    main()
