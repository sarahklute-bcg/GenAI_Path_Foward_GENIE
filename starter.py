from openai import OpenAI
import json

client = OpenAI()

# Load the output from analyze.py
with open("ai_use_case_summary.xlsx", "r") as f:
    summary = json.load(f)

# Base prompt (you can expand)
base_prompt = """
You are an AI research analyst at BCG.
Goal: Review coverage of AI use cases in different sectors and workflows.
Identify where there are gaps (sectors or workflow areas underrepresented)
and recommend directions where more use cases should be sourced.

Be concise, fact-based, and output structured JSON.
"""

user_prompt = f"""
Here is the analysis summary from my dataset:

{json.dumps(summary, indent=2)}

Please return JSON with:
- underrepresented_sectors: list of sector names with low counts
- dominant_workflows: list of workflows with high counts
- gap_recommendations: 3 specific directions to explore next
"""

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": base_prompt},
        {"role": "user", "content": user_prompt}
    ],
    response_format={"type": "json_object"}
)

print(resp.choices[0].message.content)
