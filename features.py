"""
features.py
===========
Feature engineering for the Happiness × COVID-19 project.

Transforms the wide merged DataFrame into an analysis-ready table with:
  - COVID metrics per country (3 primary measures as chosen)
  - Derived happiness features (composite indices)
  - Normalized/scaled versions for modeling

All features are documented with their business rationale.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────
# SECTION 1 — COVID METRICS
# ─────────────────────────────────────────────

def extract_covid_metrics(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer the 3 primary COVID metrics agreed upon:

    1. total_cases_apr30  — Total confirmed cases on the last date (Apr 30).
       WHY: A snapshot of cumulative pandemic severity at the end of the period.

    2. peak_daily_new_cases — Maximum single-day increase in confirmed cases.
       WHY: Captures how explosive the outbreak was at its worst moment.
           A country could have many total cases from slow spread OR from
           a sudden spike — these tell very different stories.

    3. avg_daily_growth_rate — Mean daily % increase in cases (log-based).
       WHY: Controls for country size. A country going from 1→2 cases and
           one going from 100→200 both grew 100%. Log-return is standard
           in epidemiology for comparing growth trajectories.

    Bonus features:
    4. days_to_100_cases    — Speed of early outbreak
    5. cases_log            — Log-transformed total (handles extreme skew)
    """
    df = df_merged.copy()

    # Identify date columns (all columns after the happiness features)
    happiness_cols = [
        "rank", "country", "happiness_score", "gdp_per_capita",
        "social_support", "healthy_life_expectancy", "freedom",
        "generosity", "corruption_perception"
    ]
    date_cols = [c for c in df.columns if c not in happiness_cols]
    dates     = pd.to_datetime(date_cols, format="%m/%d/%y")

    # ── Feature 1: Total cases on Apr 30 ─────
    df["total_cases_apr30"] = df[date_cols[-1]].astype(float)

    # ── Feature 2: Peak daily new cases ──────
    case_matrix = df[date_cols].astype(float).values  # shape (n_countries, n_days)
    daily_new   = np.diff(case_matrix, axis=1)         # day-over-day delta
    daily_new   = np.clip(daily_new, 0, None)          # no negatives after cleaning
    df["peak_daily_new_cases"] = daily_new.max(axis=1)

    # ── Feature 3: Average daily growth rate ─
    # Log return: ln(t+1 / t). We add 1 to avoid log(0).
    # Mean over all days gives the "typical" exponential growth factor.
    log_returns = np.diff(np.log(case_matrix + 1), axis=1)
    df["avg_daily_growth_rate"] = log_returns.mean(axis=1)

    # ── Feature 4: Days to reach 100 cases ───
    # WHY: Measures how fast the virus took hold. Countries with strong
    # public health infrastructure may delay this milestone.
    def days_to_threshold(row, threshold=100):
        for i, col in enumerate(date_cols):
            if row[col] >= threshold:
                return i
        return np.nan  # never reached threshold in dataset window

    df["days_to_100_cases"] = df.apply(days_to_threshold, axis=1)

    # ── Feature 5: Log-transformed total cases ─
    # WHY: COVID case counts span 6 orders of magnitude (1 to 1,000,000+).
    # Raw correlations with happiness would be dominated by outliers (USA, Italy).
    # Log scale reveals relationships that exist across the distribution.
    df["cases_log"] = np.log1p(df["total_cases_apr30"])

    print(f"[OK] COVID metrics engineered: 5 new features")
    return df


# ─────────────────────────────────────────────
# SECTION 2 — HAPPINESS COMPOSITE FEATURES
# ─────────────────────────────────────────────

def engineer_happiness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create composite happiness dimensions:

    1. institutional_trust_index — Avg of corruption_perception + freedom
       WHY: Measures how much citizens trust their government.
           Countries with high trust may be more likely to follow
           public health guidance during a pandemic.

    2. social_resilience_index — Avg of social_support + healthy_life_expectancy
       WHY: Captures the 'buffer' a population has against a health crisis.
           Strong social networks and healthy populations bounce back faster.

    3. economic_wellbeing_index — GDP + (1 - normalized_rank) proxy
       WHY: Wealthier nations can afford more testing, ICUs, and lockdown support.

    4. happiness_tier — Categorical grouping (Low / Medium / High / Very High)
       WHY: Enables group-level comparisons, richer than continuous correlations.
    """
    df = df.copy()

    # ── Feature 1: Institutional trust ───────
    df["institutional_trust_index"] = (
        df["corruption_perception"] + df["freedom"]
    ) / 2

    # ── Feature 2: Social resilience ─────────
    df["social_resilience_index"] = (
        df["social_support"] + df["healthy_life_expectancy"]
    ) / 2

    # ── Feature 3: Economic wellbeing ────────
    df["economic_wellbeing_index"] = df["gdp_per_capita"]  # already normalized in WHR

    # ── Feature 4: Happiness tier ─────────────
    bins   = [0, 4.5, 5.5, 6.5, 8.0]
    labels = ["Low (< 4.5)", "Medium (4.5–5.5)", "High (5.5–6.5)", "Very High (> 6.5)"]
    df["happiness_tier"] = pd.cut(
        df["happiness_score"], bins=bins, labels=labels, right=True
    )

    print(f"[OK] Happiness features engineered: 4 new features")
    print(df["happiness_tier"].value_counts())
    return df


# ─────────────────────────────────────────────
# SECTION 3 — SCALED FEATURES (FOR MODELING)
# ─────────────────────────────────────────────

def add_scaled_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Min-max scale the happiness pillar columns for use in regression/ML.
    WHY: Pillars have different native ranges; scaling puts them on equal footing
    so model coefficients are directly comparable.

    Scaled columns get a '_scaled' suffix. Original columns are preserved.
    """
    df = df.copy()

    scale_cols = [
        "gdp_per_capita", "social_support", "healthy_life_expectancy",
        "freedom", "generosity", "corruption_perception",
        "institutional_trust_index", "social_resilience_index"
    ]

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df[scale_cols])
    scaled_df     = pd.DataFrame(
        scaled_values,
        columns=[f"{c}_scaled" for c in scale_cols],
        index=df.index
    )

    df = pd.concat([df, scaled_df], axis=1)
    print(f"[OK] Scaled features added: {len(scale_cols)} columns")
    return df


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_feature_pipeline(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrate all feature engineering steps.
    Returns the fully-featured DataFrame ready for EDA and modeling.
    """
    print("=" * 55)
    print("  FEATURE ENGINEERING PIPELINE")
    print("=" * 55)

    df = extract_covid_metrics(df_merged)
    df = engineer_happiness_features(df)
    df = add_scaled_features(df)

    # Drop raw date columns from the analysis frame (keep for time-series EDA)
    happiness_cols = [
        "rank", "country", "happiness_score", "gdp_per_capita",
        "social_support", "healthy_life_expectancy", "freedom",
        "generosity", "corruption_perception"
    ]
    covid_metric_cols = [
        "total_cases_apr30", "peak_daily_new_cases",
        "avg_daily_growth_rate", "days_to_100_cases", "cases_log"
    ]
    engineered_cols = [
        "institutional_trust_index", "social_resilience_index",
        "economic_wellbeing_index", "happiness_tier"
    ]
    scaled_cols = [c for c in df.columns if c.endswith("_scaled")]

    analysis_cols = happiness_cols + covid_metric_cols + engineered_cols + scaled_cols
    df_analysis   = df[[c for c in analysis_cols if c in df.columns]].copy()

    # Save
    df_analysis.to_csv("data/processed/analysis_dataset.csv", index=False)
    print(f"\n[SAVED] data/processed/analysis_dataset.csv → {df_analysis.shape}")
    print("\n✓ Feature engineering complete.\n")

    # Also return the full df (with dates) for time-series plots
    return df, df_analysis


if __name__ == "__main__":
    import os
    os.chdir("/home/claude/happiness_covid_project")
    df_merged = pd.read_csv("data/processed/merged_dataset.csv")
    run_feature_pipeline(df_merged)
