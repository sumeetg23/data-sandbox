import pandas as pd
from pathlib import Path

# ⚙️ Paths (update as needed)
input_path = Path("../data/20250622_export_ctgov/studies.txt")
output_path = Path("../data/cleaned_trials.csv")

# Load the pipe-delimited file
df = pd.read_csv(input_path, sep="|", low_memory=False)

# Select relevant columns
columns_of_interest = [
    'nct_id',
    'brief_title',
    'study_type',
    'phase',
    'overall_status',
    'start_date',
    'completion_date',
    'primary_completion_date',
    'enrollment',
    'enrollment_type',
    'is_fda_regulated_drug',
    'is_fda_regulated_device',
    'has_dmc',
    'why_stopped',
    'plan_to_share_ipd'
]
df_clean = df[columns_of_interest].copy()

# Convert date columns
for col in ['start_date', 'completion_date', 'primary_completion_date']:
    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

# Calculate duration and start year
df_clean['duration_days'] = (df_clean['completion_date'] - df_clean['start_date']).dt.days
df_clean['start_year'] = df_clean['start_date'].dt.year

# Simplify status field
df_clean['status_simple'] = df_clean['overall_status'].replace({
    'Completed': 'Completed',
    'Recruiting': 'Active',
    'Active, not recruiting': 'Active',
    'Not yet recruiting': 'Planned',
    'Terminated': 'Terminated',
    'Withdrawn': 'Terminated',
    'Unknown status': 'Unknown'
})

# 💾 Save cleaned data to CSV
output_path.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(output_path, index=False)

print(f"Cleaned dataset saved to: {output_path.resolve()}")
