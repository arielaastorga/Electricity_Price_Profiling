# Electricity_Price_Profiling
This application retrieves, processes, and visualizes hourly marginal electricity prices by grid node from the Chilean National Electric Coordinator (CEN), enabling exploratory analysis of daily, short-term, and monthly price profiles

URL:
https://electricitypriceprofiling-bwnhzeazggiokuh9muahcq.streamlit.app/


## Main features

- Node selector to analyze a specific grid bar (`barra`).
- Date range selection with available data from **01-01-2008 to today**.
- Automatic loading of monthly CSV files based on the selected period.
- Validation to allow analysis windows of **less than 12 months**.
- KPI dashboard with three sections:
  - **General KPI**: selected period average, P50, P90, minimum and maximum values with day and hour.
  - **Solar window KPI**: average solar hours (08h-18h), average non-solar hours (19h-07h), and solar/non-solar spread.
  - **Curtailment**: hours with CMG = 0, CMG < 30, CMG < 50, and CMG < 100.
- Representative profiles using **K-Medoids**:
  - Weekday profiles.
  - Weekend profiles.
  - All-days profiles:
    - If the user selects **1 day**, the app shows the full daily curve.
    - If the user selects **more than 30 days**, the app shows **monthly representative curves**.
    - If several months are selected, the app plots **one curve per month**.
- Missing hourly values (`NaN`) are not removed by dropping the whole row; instead, they are filled using the nearest available value, prioritizing the value from the previous row.
- Dataset summary and preview displayed at the bottom of the app.

## Data source
The app works with monthly CSV files named with the pattern:

YYYYMMCmgBarras.csv

Example:
202401CmgBarras.csv
202402CmgBarras.csv
...
202412CmgBarras.csv
```

These files should be stored in:

text
data/raw/


## Expected CSV structure

The application expects the following columns:

- `fecha`
- `anio`
- `mes`
- `dia`
- `hora`
- `barra`
- `tension`
- `valor`

## How it works

1. The user selects a node and a date range.
2. The app identifies which monthly CSV files are required.
3. The selected files are loaded and merged into a single dataframe.
4. Data is cleaned and numeric columns are standardized.
5. Missing hourly values are imputed using nearby available values.
6. KPIs are calculated for the filtered node and date range.
7. Representative profiles are generated and plotted.

## Representative profile logic

### Weekday and weekend
The app builds representative daily curves separately for:
- **Weekdays**
- **Weekends**

using **K-Medoids clustering**.

### All days
The app applies the following logic:
- **1 selected day** → plot the single full-day curve.
- **More than 30 selected days** → build **monthly representative profiles**.
- **Several months selected** → one representative curve per month.

## Technologies used

- **Python**
- **Streamlit**
- **Pandas**
- **Plotly**
- **scikit-learn**
- **scikit-learn-extra**

## Run locally

Install dependencies:

bash
pip install -r requirements.txt


Run the app:

bash
streamlit run streamlit_app.py


## Notes

- The app is designed for exploratory analysis of marginal price behavior by node and time window.
- Monthly representative curves are especially useful for comparing seasonal behavior across the selected period.
- If some months do not appear in the monthly chart, it usually means the corresponding data is missing or incomplete in the source files.