import pandas as pd

# Load your Excel
df = pd.read_excel("main_data_test_200.xlsx")  # change path to your file

# 1) Count AI use cases per sector
counts = df["dem_sector_dialectica"].value_counts()
print("=== Use cases per sector ===")
print(counts)

# 2) Find rows with missing fields (example: impact_today_a or value_chain)
missing = df[df["impact_today_a"].isna() | df["value_chain"].isna()]
print("\n=== Rows needing more info ===")
print(missing[["company_name","workflow_text","dem_sector_dialectica","impact_today_a","value_chain"]])

# Optional: save results to Excel
with pd.ExcelWriter("ai_use_case_summary.xlsx") as writer:
    counts.to_excel(writer, sheet_name="Sector Counts")
    missing.to_excel(writer, sheet_name="Missing Fields", index=False)
