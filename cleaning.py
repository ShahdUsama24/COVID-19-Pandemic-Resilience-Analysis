"""
cleaning.py
===========
Data cleaning pipeline for the Happiness × COVID-19 project.

Responsibilities:
  1. Load raw datasets
  2. Clean & standardize each dataset independently
  3. Harmonize country names via fuzzy matching + manual overrides
  4. Aggregate COVID province rows → one row per country
  5. Merge into a single analysis-ready DataFrame
  6. Save processed outputs

Author : Data Analysis Project
"""

import pandas as pd
import numpy as np
from thefuzz import process, fuzz
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

RAW_HAPPINESS = "data/raw/worldwide_happiness_report.csv"
RAW_COVID     = "data/raw/covid19_Confirmed_dataset.csv"

OUT_HAPPINESS = "data/processed/happiness_clean.csv"
OUT_COVID     = "data/processed/covid_clean.csv"
OUT_MERGED    = "data/processed/merged_dataset.csv"

# Fuzzy score threshold: below this → no match (avoids false positives)
FUZZY_THRESHOLD = 85

# Manual overrides: COVID name → Happiness name
# These are cases where fuzzy matching scores < threshold
# but we know the correct mapping from domain knowledge.
MANUAL_COUNTRY_MAP = {
    "US"                           : "United States",
    "Korea, South"                 : "South Korea",
    "Taiwan*"                      : "Taiwan",
    "Burma"                        : "Myanmar",
    "Czechia"                      : "Czech Republic",
    "Cote d'Ivoire"                : "Ivory Coast",
    "Trinidad and Tobago"          : "Trinidad & Tobago",
    "Sudan"                        : "South Sudan",
    # Ships & territories → no match (will be dropped in merge)
    "Diamond Princess"             : None,
    "MS Zaandam"                   : None,
    "Holy See"                     : None,
    "West Bank and Gaza"           : None,
    "Western Sahara"               : None,
}


# ─────────────────────────────────────────────
# SECTION 1 — LOAD RAW DATA
# ─────────────────────────────────────────────

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both raw CSVs and return (df_happiness, df_covid)."""
    df_happiness = pd.read_csv(RAW_HAPPINESS)
    df_covid     = pd.read_csv(RAW_COVID)
    print(f"[LOAD] Happiness: {df_happiness.shape}  |  COVID: {df_covid.shape}")
    return df_happiness, df_covid


# ─────────────────────────────────────────────
# SECTION 2 — CLEAN HAPPINESS DATASET
# ─────────────────────────────────────────────

def clean_happiness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the World Happiness Report dataset.

    Steps:
      - Rename columns to snake_case for programmatic ease
      - Validate data types
      - Check & report any missing values
      - Flag statistical outliers (IQR method) without dropping them
        (happiness scores are legitimate — no country is truly 'wrong')
    """
    df = df.copy()

    # ── 2.1 Rename columns ───────────────────
    df.columns = [
        "rank", "country", "happiness_score",
        "gdp_per_capita", "social_support",
        "healthy_life_expectancy", "freedom",
        "generosity", "corruption_perception"
    ]

    # ── 2.2 Data types ───────────────────────
    # rank should be int, scores float — already correct, but enforce explicitly
    df["rank"] = df["rank"].astype(int)
    numeric_cols = df.columns.drop(["rank", "country"])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # ── 2.3 Missing values ───────────────────
    missing = df.isnull().sum()
    if missing.any():
        print(f"[WARN] Happiness missing values:\n{missing[missing > 0]}")
    else:
        print("[OK] Happiness: no missing values")

    # ── 2.4 Outlier flag (IQR method) ────────
    # We flag but do NOT remove — all countries are valid data points
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outlier_mask = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
        if outlier_mask.any():
            outlier_countries = df.loc[outlier_mask, "country"].tolist()
            print(f"[INFO] Outliers in '{col}': {outlier_countries}")

    print(f"[OK] Happiness cleaned → {df.shape}")
    return df


# ─────────────────────────────────────────────
# SECTION 3 — CLEAN COVID DATASET
# ─────────────────────────────────────────────

def clean_covid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the COVID-19 confirmed cases dataset.

    Steps:
      - Drop unnecessary geo columns (Lat, Long)
      - Aggregate province/state rows → one row per country (sum)
      - Parse date columns into proper datetime format (wide → tidy)
      - Validate cumulative monotonicity (cases should never decrease)
      - Rename columns for clarity
    """
    df = df.copy()

    # ── 3.1 Drop geo columns ─────────────────
    # Lat/Long are not needed for country-level analysis
    df.drop(columns=["Lat", "Long"], inplace=True)

    # ── 3.2 Aggregate provinces → country ────
    # WHY: Countries like USA (53 rows), China (33 rows) are split by province.
    # We sum ALL date columns per Country/Region to get national totals.
    date_cols = [c for c in df.columns if c not in ["Province/State", "Country/Region"]]
    df_country = (
        df.groupby("Country/Region")[date_cols]
        .sum()
        .reset_index()
    )
    df_country.rename(columns={"Country/Region": "country_raw"}, inplace=True)
    print(f"[OK] COVID aggregated: {len(df_country)} unique countries")

    # ── 3.3 Validate monotonicity ─────────────
    # Cumulative cases should never decrease day-over-day
    # (data entry errors can cause this)
    issue_countries = []
    for _, row in df_country.iterrows():
        vals = row[date_cols].values.astype(float)
        diffs = np.diff(vals)
        if (diffs < 0).any():
            issue_countries.append(row["country_raw"])

    if issue_countries:
        print(f"[WARN] Non-monotonic case counts in: {issue_countries}")
        # Fix: clip to cumulative max (forward-fill the peak)
        for country in issue_countries:
            idx = df_country[df_country["country_raw"] == country].index[0]
            vals = df_country.loc[idx, date_cols].values.astype(float)
            df_country.loc[idx, date_cols] = np.maximum.accumulate(vals)
    else:
        print("[OK] COVID: All cumulative series are monotonic")

    print(f"[OK] COVID cleaned → {df_country.shape}")
    return df_country


# ─────────────────────────────────────────────
# SECTION 4 — HARMONIZE COUNTRY NAMES
# ─────────────────────────────────────────────

def harmonize_country_names(
    df_covid: pd.DataFrame,
    happiness_countries: list[str]
) -> pd.DataFrame:
    """
    Map COVID country names → Happiness country names using:
      1. Manual override dict (highest priority, domain knowledge)
      2. Fuzzy matching with token_sort_ratio (handles word reordering)
      3. Exact match fallback

    Returns df_covid with a new column 'country' (harmonized name).
    Countries that cannot be matched are marked as None and will be
    dropped during the merge step.
    """
    df = df_covid.copy()

    def resolve_name(raw_name: str) -> str | None:
        # Priority 1: manual override
        if raw_name in MANUAL_COUNTRY_MAP:
            return MANUAL_COUNTRY_MAP[raw_name]  # may be None (intentional drop)

        # Priority 2: exact match
        if raw_name in happiness_countries:
            return raw_name

        # Priority 3: fuzzy match
        match, score = process.extractOne(
            raw_name, happiness_countries, scorer=fuzz.token_sort_ratio
        )
        if score >= FUZZY_THRESHOLD:
            return match

        # No reliable match found
        print(f"  [NO MATCH] '{raw_name}' (best fuzzy: '{match}' @ {score})")
        return None

    df["country"] = df["country_raw"].apply(resolve_name)

    matched   = df["country"].notna().sum()
    unmatched = df["country"].isna().sum()
    print(f"[OK] Name harmonization: {matched} matched, {unmatched} unmatched (will be dropped in merge)")
    return df


# ─────────────────────────────────────────────
# SECTION 5 — MERGE DATASETS
# ─────────────────────────────────────────────

def merge_datasets(
    df_happiness: pd.DataFrame,
    df_covid: pd.DataFrame
) -> pd.DataFrame:
    """
    Inner-join happiness and COVID on harmonized country name.

    WHY inner join: We only want countries present in BOTH datasets.
    Countries only in one dataset cannot contribute to correlation analysis.

    Returns the merged wide-format DataFrame (one row per country).
    """
    # Drop unmatched COVID countries
    df_covid_matched = df_covid.dropna(subset=["country"])

    merged = pd.merge(
        df_happiness,
        df_covid_matched.drop(columns=["country_raw"]),
        on="country",
        how="inner"
    )

    print(f"[OK] Merged dataset: {merged.shape[0]} countries × {merged.shape[1]} columns")

    # Coverage report
    happy_only  = set(df_happiness["country"]) - set(merged["country"])
    covid_only  = set(df_covid_matched["country"]) - set(merged["country"])
    print(f"     Countries only in Happiness (not in merged): {len(happy_only)}")
    print(f"     Countries only in COVID     (not in merged): {len(covid_only)}")
    if happy_only:
        print(f"     Sample (happiness only): {sorted(happy_only)[:10]}")

    return merged


# ─────────────────────────────────────────────
# SECTION 6 — SAVE OUTPUTS
# ─────────────────────────────────────────────

def save_processed(
    df_happiness: pd.DataFrame,
    df_covid: pd.DataFrame,
    df_merged: pd.DataFrame
) -> None:
    """Save all three processed DataFrames to CSV."""
    df_happiness.to_csv(OUT_HAPPINESS, index=False)
    df_covid.to_csv(OUT_COVID, index=False)
    df_merged.to_csv(OUT_MERGED, index=False)
    print(f"\n[SAVED] {OUT_HAPPINESS}")
    print(f"[SAVED] {OUT_COVID}")
    print(f"[SAVED] {OUT_MERGED}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_cleaning_pipeline() -> dict[str, pd.DataFrame]:
    """
    Orchestrate the full cleaning pipeline end-to-end.
    Returns a dict with keys: 'happiness', 'covid', 'merged'.
    """
    print("=" * 55)
    print("  CLEANING PIPELINE — Happiness × COVID-19")
    print("=" * 55)

    # Step 1 — Load
    df_happiness_raw, df_covid_raw = load_raw_data()

    # Step 2 — Clean each dataset independently
    print("\n── Happiness ──────────────────────────────────")
    df_happiness = clean_happiness(df_happiness_raw)

    print("\n── COVID-19 ───────────────────────────────────")
    df_covid = clean_covid(df_covid_raw)

    # Step 3 — Harmonize names
    print("\n── Country Name Harmonization ─────────────────")
    happiness_countries = df_happiness["country"].tolist()
    df_covid = harmonize_country_names(df_covid, happiness_countries)

    # Step 4 — Merge
    print("\n── Merging ─────────────────────────────────────")
    df_merged = merge_datasets(df_happiness, df_covid)

    # Step 5 — Save
    print("\n── Saving ──────────────────────────────────────")
    save_processed(df_happiness, df_covid, df_merged)

    print("\n✓ Pipeline complete.\n")
    return {"happiness": df_happiness, "covid": df_covid, "merged": df_merged}


if __name__ == "__main__":
    import os
    os.chdir("/home/claude/happiness_covid_project")
    run_cleaning_pipeline()
