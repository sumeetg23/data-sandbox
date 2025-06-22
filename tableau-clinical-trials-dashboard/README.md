# Clinical Trials Explorer – Tableau Dashboard

A data visualization that explores clinical trial metrics using data from ClinicalTrials.gov.

---

## Project Overview

This project analyzes and visualizes global clinical trial data, focusing on:
- Trial distribution across phases and statuses
- Start-date trends over time
- Trial durations
- FDA regulation and IPD-sharing insights

The goal is to provide insights into the clinical trial landscape using Tableau.

---

## File Structure

      tableau-clinical-trials-dashboard/
      ├── data/
      │   ├── 20250622_export_ctgov/          ← Raw ClinicalTrials.gov export (ignored in repo)
      │   └── cleaned_trials.csv              ← Cleaned dataset for Tableau
      ├── scripts/
      │   └── clean_clinical_trials.py        ← Python data‑prep script
      ├── dashboard/
      │   └── clinical_trials_dashboard.twbx ← Tableau packaged workbook
      ├── screenshots/
      │   └── dashboard_preview.png           ← Static preview of the dashboard
      └── README.md                           ← This file
  

## Data Cleaning & Preparation

- **Source**: ClinicalTrials.gov export (dated 2025-06-22)  
- **Cleaning process**:  
  - Converted pipe-delimited raw data into structured CSV  
  - Selected key fields including phase, status, dates, enrollment, FDA regulation, and IPD-sharing  
  - Parsed dates, calculated trial duration and start year  
  - Simplified status (e.g., "Active", "Completed")  
- **Script**: `clean_clinical_trials.py` (requires `pandas`)

### How to Regenerate Data

1. **Run the data preparation script** (if you want to rebuild):

cd tableau-clinical-trials-dashboard/scripts
python3 clean_clinical_trials.py  (This creates data/cleaned_trials.csv)

2. Open the Tableau workbook:

Navigate to dashboard/clinical_trials_dashboard.twbx (Explore visuals or modify them as needed)

Dashboard from Tableau Public Available here: https://public.tableau.com/app/profile/sumeet.gupta/viz/clinical_trials_dashboard/Dashboard1


