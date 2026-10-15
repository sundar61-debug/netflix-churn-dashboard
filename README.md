# Why Do Subscribers Walk Away? — Netflix Churn Dashboard

An interactive Streamlit dashboard that answers one question:

> **What separates Netflix customers who cancel from those who stay, and can a simple model predict who is at risk?**

**Live app:** `https://netflix-churn-dashboard-vfpfnnu5uwijti4awcjmur.streamlit.app/`
**Repo:** `https://github.com/sundar61-debug/netflix-churn-dashboard`

---

## The finding

- **Engagement is the strongest signal.** Customers who watch very little and haven't logged in for weeks churn the most (watch hours correlate about -0.48 with churn, days since last login about +0.47).
- **Plan and payment method matter too.** Basic-plan customers churn at ~62% vs. ~44% for Premium. Gift Card and Crypto payers churn at ~58–60% vs. ~44% for Credit/Debit Card payers.
- **Region, device, genre and gender barely matter** (all within a few points of the 50% average).
- **A logistic regression predicts churn well:** ~88% test accuracy vs. ~50% for the constant baseline.

## What's inside (maps to the project rubric)

**| Requirement | Where |**
| Data cleaning, missing-value rule, `.groupby()`, derived columns | `load_and_clean()` in `app.py`, "Data & cleaning" tab |
| Faithfulness & Granularity | Section below + "Data & cleaning" tab |
| 3+ visualizations (bar, histogram, scatter, heatmap) | "Explore" tab |
| One-hot encoding with a stated reference level | `train_models()`, "Model" tab |
| Constant model + Logistic Regression, predictions vs. actual | "Model" tab |
| 2+ widgets and `st.columns` layout | Sidebar filters, selectbox/radio, "Predict a customer" tab, KPI row |

## Faithfulness & Granularity

**What is one row?** One subscriber account. `customer_id` is unique, so there are no repeated customers. Each row is a single snapshot of that customer's plan, payment method, device, usage and whether they churned.

**What is the scope?** 5,000 customers across 6 regions, ages 18–70, with 14 original columns. There is **no date column**, so we can't see *when* anyone churned or how behavior changed over time. The data is also unusually balanced (about 50% churn, similar group sizes everywhere), which suggests it is curated or simulated rather than a raw production log. The findings describe patterns in *this dataset*, not claims about the real Netflix.

**Strange values, inconsistencies, duplicates:**
- 0 duplicate rows and 0 duplicate customer IDs.
- 10 rows (0.2%) show more than 24 hours watched *per day* in `avg_watch_time_per_day`, which is impossible. These were dropped.
- `avg_watch_time_per_day` is usually a small fraction of `watch_hours` but sometimes equals it exactly, so it is excluded from the model.
- `monthly_fee` has exactly one price per plan (Basic 8.99, Standard 13.99, Premium 17.99), so it duplicates `subscription_type` and is excluded to avoid collinearity.
- `watch_hours` is right-skewed (max about 110). Heavy users look real, so they were kept.

## Missing values

Every column has **0% missing**, so nothing needed imputing or dropping. The code still applies an explicit rule so the decision is documented and reproducible: **under 5% missing → impute** (median for numbers, "Unknown" for text); **5% or more → drop the column**. In a dataset with real gaps, I'd also check whether missingness correlates with `churned` before imputing.

## Modeling

- **Target:** `churned` (1 = cancelled). **Split:** 80/20 stratified, `random_state=42` (3,992 train / 998 test after cleaning).
- **Features:** watch hours, days since last login, number of profiles, age, plus one-hot encoded **subscription plan** and **payment method**.
- **Reference levels dropped (`drop_first=True`):** plan → **Basic**, payment method → **Credit Card**.
- **Train/test alignment:** the test set is re-indexed to the training columns (`reindex(columns=..., fill_value=0)`), so a category missing from one split can't break the model.
- **Plain-language coefficient example:** holding everything else constant, a **Premium** customer has about **0.11×** the odds of churning compared with a **Basic** customer. A customer paying by **Gift Card** has about **5×** the odds of a **Credit Card** payer.

**| Model | Test accuracy | Test log loss |**

| Constant (predict base rate) | ~50% | 0.693 |
| Logistic, numeric features only | ~87% | 0.310 |
| Logistic, + one-hot plan & payment | ~88% | 0.251 |

**Did the encoded feature help?** Yes. Test log loss dropped about 19% and accuracy improved by a point and a half.

**Evaluation observation:** predicted probabilities line up closely with actual churn rates. Mistakes cluster among customers who get a probability near 50%, where the model is genuinely unsure.

## Run it locally

```bash
git clone <your-repo-url>
cd netflix-churn-dashboard
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Tech

Python · pandas · NumPy · matplotlib · seaborn · scikit-learn · Streamlit
