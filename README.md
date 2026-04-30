#  Happiness & COVID-19 Pandemic Resilience Analysis

> *Does national wellbeing predict pandemic outcomes? A cross-country data analysis across 151 nations.*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=flat&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-4C72B0?style=flat)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

##  Project Overview

This project investigates whether a country's happiness and social infrastructure — as measured by the **2019 World Happiness Report** — correlates with COVID-19 spread patterns during the early pandemic phase (January–April 2020).

The analysis spans **151 countries**, engineers **9 custom features**, and produces **6 advanced visualizations** — going beyond surface-level correlations to challenge assumptions about what "more cases" actually means.

---

## Business Questions

1. Do happier countries have fewer or slower COVID-19 cases?
2. Which happiness pillar (GDP, social support, freedom…) correlates most with outbreak severity?
3. How do happiness tiers compare in outbreak speed and trajectory?
4. Are there resilient outliers worth studying?
5. Can happiness indicators predict COVID outcomes?

---



##  Datasets

| Dataset | Source | Shape | Description |
|---|---|---|---|
| World Happiness Report 2019 | [Kaggle](https://www.kaggle.com/datasets/unsdsn/world-happiness) | 156 × 9 | Country happiness scores & 6 pillar scores |
| COVID-19 Confirmed Cases | [Johns Hopkins / Kaggle](https://www.kaggle.com/datasets/imdevskp/corona-virus-report) | 266 × 104 | Daily cumulative confirmed cases by country/province |

---

## Methodology

### Data Cleaning
- **Province aggregation**: Countries split by province (USA, China, Canada, UK, France, Australia, Netherlands, Denmark) were summed to a single country-level row
- **Country name harmonization**: Fuzzy matching (`thefuzz`, token sort ratio, threshold = 85) + 8 manual overrides to join datasets across inconsistent naming conventions
- **Monotonicity correction**: 10 countries had non-monotonic cumulative case series (data entry errors) — fixed via cumulative maximum forward-fill
- **Result**: 151 matched countries after inner join (36 COVID-only countries dropped, 6 happiness-only countries dropped)

### Feature Engineering

**COVID Metrics (3 primary + 2 derived):**

| Feature | Description | Why it matters |
|---|---|---|
| `total_cases_apr30` | Cumulative confirmed cases on Apr 30 | Snapshot of outbreak magnitude |
| `avg_daily_growth_rate` | Mean log return of daily cases | Controls for testing capacity bias |
| `peak_daily_new_cases` | Maximum single-day case increase | Captures outbreak explosiveness |
| `days_to_100_cases` | Days from Jan 22 to reach 100 cases | Measures early spread speed |
| `cases_log` | Log₁ₑ(total cases + 1) | Handles extreme right skew |

**Happiness Composites:**

| Feature | Formula | Rationale |
|---|---|---|
| `institutional_trust_index` | (freedom + corruption_perception) / 2 | Trust in government → compliance with health measures |
| `social_resilience_index` | (social_support + healthy_life_expectancy) / 2 | Population buffer against health crises |
| `happiness_tier` | Cut into 4 bins | Enables group-level analysis |

---

## Key Visualizations

| Figure | Type | Key Insight |
|---|---|---|
| Fig 01 | KDE + Boxplot | Social Support is the most differentiating happiness pillar globally |
<img width="1807" height="636" alt="fig_01_happiness_distribution" src="https://github.com/user-attachments/assets/6f04d562-4bcd-4f9b-b1ce-e89f1dff404a" />

| Fig 02 | Time-series by tier | All tiers show the same exponential surge from mid-March — tier doesn't change trajectory shape |
<img width="2061" height="1280" alt="fig_02_trajectories_by_tier" src="https://github.com/user-attachments/assets/74d33454-59dc-43bc-a248-9ef73260c76e" />

| Fig 03 | Scatter + Regression | Happier nations reached 100 cases faster (connectivity effect), not slower |
<img width="2326" height="767" alt="fig_03_scatter_regression" src="https://github.com/user-attachments/assets/a22ae4f4-e69f-4070-84ee-fedd80e871bf" />

| Fig 04 | Spearman Heatmap | GDP, Social Support & Life Expectancy form a tight development cluster (r > 0.76) |
<img width="1696" height="1416" alt="fig_04_correlation_heatmap" src="https://github.com/user-attachments/assets/c79d59fc-94fa-41c3-a408-1034e6ae1f78" />

| Fig 05 | Violin + Strip | Growth rate distributions are nearly identical across tiers — happiness alone is a weak predictor |
<img width="2327" height="898" alt="fig_05_violin_tier" src="https://github.com/user-attachments/assets/7d7ab189-8e97-444d-b053-f2e61955ddcb" />

| Fig 06 | Horizontal Bar | Top happiest nations grew faster than bottom — but bottom nations likely reflect undertesting |
<img width="2064" height="898" alt="fig_06_top_vs_bottom" src="https://github.com/user-attachments/assets/f60eb2be-d38d-45e6-bdc3-ae231e03618e" />


---

##  Key Findings

**1. Testing capacity, not happiness, drives raw case counts.**
Happier/wealthier countries reported more cases (r = 0.59) — because they tested more and reported more reliably. Raw totals are a biased metric for cross-country comparison.

**2. Growth rate is the fairest pandemic metric.**
Unlike total cases, average daily growth rate does not depend on testing volume. Yet even here, happiness tier shows only moderate predictive power — other factors (geography, density, policy timing) dominate.

**3. Wealthier nations were hit first.**
Days to 100 cases correlates strongly negatively with GDP (r = −0.78) and social support (r = −0.51). International connectivity, not vulnerability, explains this.

**4. Generosity is analytically independent.**
The Generosity pillar has near-zero correlation with all other happiness pillars and all COVID metrics — it measures something orthogonal to national development.

**5. High-trust institutions may offer mild protection.**
Nordic countries (Finland, Iceland, New Zealand) cluster at the low end of growth rates within the Very High happiness tier — consistent with the hypothesis that citizens in high-trust societies are more likely to follow public health guidance.

---

## Limitations & Caveats

- **No population normalization** — cases per capita would be a stronger metric; raw totals heavily favor large countries
- **No testing rate data** — the single largest confounder in this analysis
- **Static happiness snapshot** — 2019 WHR may not reflect conditions in early 2020
- **100-day window only** — analysis covers early outbreak phase; later pandemic dynamics may differ
- **Low-happiness country data quality** — many low-income countries almost certainly undertested, making their low case counts unreliable


---

## Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
scikit-learn>=1.3
thefuzz>=0.19
python-Levenshtein>=0.21
jupyter>=1.0
nbformat>=5.9
```

---

##  Tools & Libraries

| Tool | Purpose |
|---|---|
| Pandas | Data loading, cleaning, aggregation |
| NumPy | Vectorized COVID metric computation |
| Matplotlib / Seaborn | All visualizations |
| SciPy | Pearson/Spearman correlation, regression |
| Scikit-learn | MinMaxScaler for feature normalization |
| thefuzz | Fuzzy country name matching |
| Jupyter | Interactive notebook environment |

---

## Author

**[Shahd Usama]**
Data Analyst | Python · SQL · Machine Learning

[![[LinkedIn](https://www.linkedin.com/in/shahdusama/)](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![[GitHub](https://github.com/ShahdUsama24)](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/YOUR_USERNAME)


*Data sources are publicly available on Kaggle. This project is for educational and portfolio purposes.*
