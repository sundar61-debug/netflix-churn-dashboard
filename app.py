"""
Why Do Subscribers Walk Away?  ·  Netflix churn dashboard
Central question: what separates customers who cancel from customers who stay,
and can a simple model predict who is at risk?

Pipeline: clean (pandas) -> explore (EDA) -> visualize (matplotlib/seaborn)
          -> one-hot encode -> Constant model + Logistic Regression -> Streamlit
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# 1. PAGE SETUP + THEME
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Why Subscribers Leave · Netflix Churn",
    page_icon="🎬",
    layout="wide",
)

RED, GREEN, GREY = "#E50914", "#46d369", "#8a8a95"
BG, CARD, LINE, TXT = "#0b0b0f", "#16161d", "#2a2a33", "#f5f5f1"

plt.rcParams.update({
    "figure.facecolor": CARD, "axes.facecolor": CARD, "savefig.facecolor": CARD,
    "axes.edgecolor": LINE, "axes.labelcolor": "#c9c9d1", "text.color": TXT,
    "xtick.color": GREY, "ytick.color": GREY, "grid.color": LINE,
    "axes.grid": True, "axes.axisbelow": True, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
    "legend.labelcolor": TXT,
})
CHURN_PALETTE = {"Stayed": GREEN, "Churned": RED}
HEAT = LinearSegmentedColormap.from_list("nf", [GREEN, CARD, RED])

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; max-width: 1250px; }
header[data-testid="stHeader"] { background: transparent; }

.hero {
  background: radial-gradient(900px 380px at 0% 0%, rgba(229,9,20,.38), transparent 60%),
              linear-gradient(135deg, #16161d, #0b0b0f);
  border: 1px solid #23232c; border-radius: 22px;
  padding: 2.3rem 2.6rem; margin-bottom: 1.3rem;
}
.hero .tag { color: #E50914; font-weight: 700; letter-spacing: .25em; font-size: .75rem; }
.hero h1 { font-family: 'Bebas Neue', sans-serif; letter-spacing: .04em;
           font-size: 3.8rem; margin: .2rem 0 0; color: #f5f5f1; line-height: 1; }
.hero h1 span { color: #E50914; }
.hero p { color: #a5a5b0; font-size: 1.05rem; margin: .7rem 0 0; max-width: 780px; }

.kpi { background: #16161d; border: 1px solid #23232c; border-radius: 16px;
       padding: 1rem 1.25rem; transition: transform .15s ease, border-color .15s ease; }
.kpi:hover { transform: translateY(-3px); border-color: #E50914; }
.kpi-label { text-transform: uppercase; letter-spacing: .14em; font-size: .68rem; color: #8a8a95; }
.kpi-value { font-family: 'Bebas Neue', sans-serif; font-size: 2.6rem; color: #f5f5f1; line-height: 1.1; }
.kpi-sub { font-size: .8rem; color: #8a8a95; }

.callout { border-left: 4px solid #E50914; background: #16161d; color: #e6e6ea;
           border-radius: 0 14px 14px 0; padding: 1rem 1.4rem; margin: 1.1rem 0 1.3rem; }
.callout b { color: #fff; }
.note { background: #12121a; border: 1px dashed #2f2f3a; border-radius: 12px;
        padding: .8rem 1.1rem; color: #b8b8c2; font-size: .93rem; }

.section-title { font-family: 'Bebas Neue', sans-serif; font-size: 1.9rem;
                 letter-spacing: .05em; margin: 1.4rem 0 .2rem; }
.section-title span { color: #E50914; }
.section-sub { color: #8a8a95; margin-bottom: .8rem; font-size: .95rem; }

.gauge { height: 18px; border-radius: 99px; background: #23232c; overflow: hidden; margin: .5rem 0 .3rem; }
.gauge-fill { height: 100%; border-radius: 99px; transition: width .4s ease; }
.big-prob { font-family: 'Bebas Neue', sans-serif; font-size: 5rem; line-height: 1; }

.stTabs [data-baseweb="tab-list"] { gap: .45rem; }
.stTabs [data-baseweb="tab"] { background: #16161d; border-radius: 10px; padding: .55rem 1.1rem; }
.stTabs [aria-selected="true"] { background: #E50914 !important; color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
section[data-testid="stSidebar"] { border-right: 1px solid #23232c; }
</style>
""",
    unsafe_allow_html=True,
)


def section(title: str, sub: str = "", accent: str = ""):
    """Consistent section header."""
    st.markdown(
        f"<div class='section-title'>{title} <span>{accent}</span></div>"
        f"<div class='section-sub'>{sub}</div>",
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, sub: str = ""):
    st.markdown(
        f"<div class='kpi'><div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div><div class='kpi-sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def show(fig):
    """Embed a matplotlib figure via st.pyplot, then free it."""
    st.pyplot(fig)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 2. DATA CLEANING (Pandas)
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / "data" / "netflix_customer_churn.csv"

PLAN_ORDER = ["Basic", "Standard", "Premium"]
TIER_ORDER = ["Light", "Casual", "Regular", "Binge"]
AGE_ORDER = ["18-29", "30-44", "45-59", "60+"]
RECENCY_ORDER = ["This week", "This month", "Gone quiet"]
CAT_COLS = ["subscription_type", "payment_method", "region", "device",
            "favorite_genre", "gender"]


@st.cache_data(show_spinner=False)
def load_and_clean():
    raw = pd.read_csv(DATA_PATH)

    # ---- audit the raw data BEFORE touching it
    audit = {
        "raw_rows": len(raw),
        "missing_pct": raw.isna().mean().mul(100).round(2),
        "dup_rows": int(raw.duplicated().sum()),
        "dup_ids": int(raw["customer_id"].duplicated().sum()),
    }
    df = raw.copy()

    # ---- text hygiene (.str): stray whitespace would silently split categories
    for c in CAT_COLS:
        df[c] = df[c].str.strip()

    # ---- missing values: rule-based so the decision is explicit and justified
    #   <5% of rows missing  -> impute (median for numbers, "Unknown" for text)
    #   >=5% of rows missing -> drop the column (too much invented data)
    # (This dataset has 0% missing, so nothing triggers, but the rule is live.)
    imputed, dropped = [], []
    for col in df.columns[df.isna().any()]:
        pct = df[col].isna().mean() * 100
        if pct < 5:
            fill = df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else "Unknown"
            df[col] = df[col].fillna(fill)
            imputed.append(col)
        else:
            df = df.drop(columns=col)
            dropped.append(col)
    audit["imputed"], audit["dropped_cols"] = imputed, dropped

    # ---- duplicates + impossible values
    df = df.drop_duplicates(subset="customer_id")
    impossible = df["avg_watch_time_per_day"] > 24  # >24 hrs watched in one day
    audit["impossible_rows"] = int(impossible.sum())
    df = df.loc[~impossible].reset_index(drop=True)
    audit["clean_rows"] = len(df)

    # ---- derived / engineered columns
    df["short_id"] = df["customer_id"].str[:8]                              # .str
    df["churn_label"] = df["churned"].map({0: "Stayed", 1: "Churned"})
    df["engagement_tier"] = pd.cut(                                         # pd.cut
        df["watch_hours"], bins=[-np.inf, 3, 8, 16, np.inf], labels=TIER_ORDER
    ).astype(str)
    df["age_group"] = pd.cut(
        df["age"], bins=[17, 29, 44, 59, 70], labels=AGE_ORDER
    ).astype(str)
    df["login_recency"] = df["last_login_days"].apply(                      # .apply
        lambda d: "This week" if d <= 7 else ("This month" if d <= 30 else "Gone quiet")
    )
    return df, audit


df, audit = load_and_clean()

# ──────────────────────────────────────────────────────────────────────────────
# 3. MODELS: Constant model vs Logistic Regression (one-hot encoded features)
# ──────────────────────────────────────────────────────────────────────────────
NUM_FEATS = ["watch_hours", "last_login_days", "number_of_profiles", "age"]
ENC_FEATS = ["subscription_type", "payment_method"]  # one-hot encoded

FEAT_LABEL = {
    "watch_hours": "Watch hours", "last_login_days": "Days since last login",
    "number_of_profiles": "Profiles on account", "age": "Age",
}
FIELD_NAME = {"subscription_type": "plan", "payment_method": "payment"}


def pretty(col: str) -> str:
    if col in FEAT_LABEL:
        return FEAT_LABEL[col]
    for f in ENC_FEATS:
        if col.startswith(f + "_"):
            return f"{col[len(f) + 1:]} ({FIELD_NAME[f]})"
    return col


@st.cache_resource(show_spinner="Training models…")
def train_models(data: pd.DataFrame):
    X, y = data[NUM_FEATS + ENC_FEATS], data["churned"]
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # One-hot encode. drop_first=True drops the alphabetically-first level of each
    # column; that level becomes the REFERENCE level every coefficient is compared to.
    Xtr = pd.get_dummies(Xtr_raw, columns=ENC_FEATS, drop_first=True, dtype=float)
    Xte = pd.get_dummies(Xte_raw, columns=ENC_FEATS, drop_first=True, dtype=float)
    # Guarantee identical columns in train and test (rare category in only one split)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
    refs = {c: sorted(Xtr_raw[c].unique())[0] for c in ENC_FEATS}

    # C=1e6 ≈ no regularization, so coefficients are plain logistic-regression ones
    m_num = LogisticRegression(C=1e6, max_iter=5000).fit(Xtr[NUM_FEATS], ytr)
    m_full = LogisticRegression(C=1e6, max_iter=5000).fit(Xtr, ytr)

    p_const = float(ytr.mean())
    preds = {
        "Constant model": np.full(len(yte), p_const),
        "Logistic · numeric only": m_num.predict_proba(Xte[NUM_FEATS])[:, 1],
        "Logistic · + one-hot plan & payment": m_full.predict_proba(Xte)[:, 1],
    }
    metrics = pd.DataFrame({
        name: {
            "Accuracy": accuracy_score(yte, p >= 0.5),
            "Log loss": log_loss(yte, p),
            "ROC AUC": roc_auc_score(yte, p),
        }
        for name, p in preds.items()
    }).T

    return dict(m_full=m_full, cols=list(Xtr.columns), Xtr=Xtr, ytr=ytr, yte=yte,
                p_full=preds["Logistic · + one-hot plan & payment"],
                metrics=metrics, refs=refs, p_const=p_const,
                n_train=len(Xtr), n_test=len(Xte))


M = train_models(df)
model, COLS = M["m_full"], M["cols"]


def build_rows(n=1, **num_and_cat) -> pd.DataFrame:
    """Create n model-ready rows (columns aligned to training) from raw inputs."""
    row = {c: 0.0 for c in COLS}
    for k in NUM_FEATS:
        row[k] = float(num_and_cat[k])
    for f in ENC_FEATS:
        key = f"{f}_{num_and_cat[f]}"
        if key in row:  # reference level has no column -> stays all zeros
            row[key] = 1.0
    return pd.DataFrame([row] * n)[COLS]


# ──────────────────────────────────────────────────────────────────────────────
# 4. SIDEBAR  (interactive widgets #1-#3)
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Filter the data")
    st.caption("These filters drive the **Explore** tab.")
    plans = st.multiselect("Subscription plan", PLAN_ORDER, default=PLAN_ORDER)
    age_lo, age_hi = st.slider("Age range", 18, 70, (18, 70))
    region = st.selectbox("Region", ["All regions"] + sorted(df["region"].unique()))
    st.divider()
    st.caption("Built with pandas · seaborn · scikit-learn · Streamlit")

fdf = df[df["subscription_type"].isin(plans) & df["age"].between(age_lo, age_hi)]
if region != "All regions":
    fdf = fdf[fdf["region"] == region]

# ──────────────────────────────────────────────────────────────────────────────
# 5. HERO + KPIs  (st.columns side-by-side layout)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
  <div class="tag">CUSTOMER CHURN · 5,000 SUBSCRIBERS</div>
  <h1>Why do subscribers <span>walk away?</span></h1>
  <p>Explore who cancels and who stays, then see how well a simple model can
  predict it. Use the sidebar to slice the data and the last tab to test a customer yourself.</p>
</div>
""",
    unsafe_allow_html=True,
)

if fdf.empty:
    st.warning("No customers match those filters. Widen the sidebar selections.")
    st.stop()

overall_churn = df["churned"].mean()
k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi("Customers in view", f"{len(fdf):,}", f"of {len(df):,} total")
with k2:
    d = (fdf["churned"].mean() - overall_churn) * 100
    kpi("Churn rate", f"{fdf['churned'].mean():.1%}", f"{d:+.1f} pts vs. all customers")
with k3:
    kpi("Avg. watch hours", f"{fdf['watch_hours'].mean():.1f}",
        f"stayed {fdf.loc[fdf.churned == 0, 'watch_hours'].mean():.1f} · "
        f"churned {fdf.loc[fdf.churned == 1, 'watch_hours'].mean():.1f}")
with k4:
    kpi("Avg. days since login", f"{fdf['last_login_days'].mean():.0f}",
        f"stayed {fdf.loc[fdf.churned == 0, 'last_login_days'].mean():.0f} · "
        f"churned {fdf.loc[fdf.churned == 1, 'last_login_days'].mean():.0f}")

# headline finding, computed from the data so it is never stale
plan_rate = df.groupby("subscription_type")["churned"].mean()
pay_rate = df.groupby("payment_method")["churned"].mean()
card_rate = df.loc[df["payment_method"].isin(["Credit Card", "Debit Card"]), "churned"].mean()
risky_rate = df.loc[df["payment_method"].isin(["Crypto", "Gift Card"]), "churned"].mean()
st.markdown(
    f"""<div class='callout'><b>The short answer:</b> engagement and how people pay.
Basic-plan customers churn at <b>{plan_rate['Basic']:.0%}</b> versus
<b>{plan_rate['Premium']:.0%}</b> on Premium. Gift Card and Crypto payers churn at
<b>{risky_rate:.0%}</b> versus <b>{card_rate:.0%}</b> for card payers. Watch time and
login recency are the strongest numeric signals. Region, device and genre barely matter.</div>""",
    unsafe_allow_html=True,
)

tab_explore, tab_model, tab_predict, tab_data = st.tabs(
    ["📊  Explore", "🤖  Model", "🔮  Predict a customer", "🧹  Data & cleaning"]
)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 · EXPLORE  (4 visualizations)
# ──────────────────────────────────────────────────────────────────────────────
with tab_explore:
    section("Who leaves?", "Every chart below respects your sidebar filters.", "")
    c1, c2 = st.columns(2)

    # ---- (1) CATEGORICAL BAR PLOT + interactive widget #4
    with c1:
        by_options = {
            "Subscription plan": "subscription_type", "Payment method": "payment_method",
            "Region": "region", "Device": "device", "Favorite genre": "favorite_genre",
            "Gender": "gender", "Age group": "age_group",
            "Engagement tier": "engagement_tier", "Login recency": "login_recency",
        }
        label = st.selectbox("Break churn down by…", list(by_options), index=1)
        col = by_options[label]
        g = (fdf.groupby(col)["churned"].agg(rate="mean", n="size")
                .sort_values("rate", ascending=False))
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [RED if r > fdf["churned"].mean() else "#4a4a57" for r in g["rate"]]
        bars = ax.bar(range(len(g)), g["rate"], color=colors, width=0.62)
        for b, r in zip(bars, g["rate"]):
            ax.text(b.get_x() + b.get_width() / 2, r + 0.012, f"{r:.0%}",
                    ha="center", fontsize=9, color=TXT, fontweight="bold")
        ax.axhline(fdf["churned"].mean(), color=GREY, ls="--", lw=1)
        ax.text(len(g) - 0.5, fdf["churned"].mean() + 0.01, "average", color=GREY,
                fontsize=8, ha="right")
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels([f"{i}\n(n={n:,})" for i, n in zip(g.index, g["n"])], fontsize=8)
        ax.set_ylim(0, min(1.05, g["rate"].max() + 0.12))
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_ylabel("Churn rate")
        ax.set_title(f"Churn rate by {label.lower()}")
        ax.grid(axis="x", visible=False)
        show(fig)

    # ---- (2) QUANTITATIVE HISTOGRAM + interactive widget #5
    with c2:
        num_options = {"Watch hours": "watch_hours", "Days since last login": "last_login_days",
                       "Age": "age", "Profiles on account": "number_of_profiles"}
        nlabel = st.radio("Distribution of…", list(num_options), horizontal=True)
        ncol = num_options[nlabel]
        fig, ax = plt.subplots(figsize=(6, 4))
        upper = fdf[ncol].quantile(0.99)  # clip extreme tail so the shape is readable
        sns.histplot(data=fdf[fdf[ncol] <= upper], x=ncol, hue="churn_label",
                     hue_order=["Stayed", "Churned"], palette=CHURN_PALETTE,
                     bins=28, stat="density", common_norm=False, element="step",
                     alpha=0.35, linewidth=1.6, ax=ax)
        ax.set_xlabel(nlabel)
        ax.set_ylabel("Density")
        ax.set_title(f"{nlabel}: stayed vs. churned")
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("")
            sns.move_legend(ax, "upper right")
        show(fig)

    c3, c4 = st.columns(2)

    # ---- (3) RELATIONSHIP: scatter
    with c3:
        sample = fdf.sample(min(1500, len(fdf)), random_state=7)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=sample, x="watch_hours", y="last_login_days",
                        hue="churn_label", hue_order=["Stayed", "Churned"],
                        palette=CHURN_PALETTE, s=16, alpha=0.55, linewidth=0, ax=ax)
        ax.set_xlim(0, fdf["watch_hours"].quantile(0.99))
        ax.set_xlabel("Watch hours")
        ax.set_ylabel("Days since last login")
        ax.set_title("Quiet and absent = at risk")
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("")
        show(fig)

    # ---- (4) CORRELATION HEATMAP
    with c4:
        corr_cols = ["age", "watch_hours", "last_login_days", "number_of_profiles",
                     "monthly_fee", "avg_watch_time_per_day", "churned"]
        corr = fdf[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, mask=np.eye(len(corr), dtype=bool), annot=True, fmt=".2f",
                    cmap=HEAT, center=0, vmin=-1, vmax=1,
                    linewidths=1, linecolor=CARD, cbar=False, annot_kws={"size": 8},
                    xticklabels=["age", "watch hrs", "login days", "profiles", "fee",
                                 "avg/day", "churned"],
                    yticklabels=["age", "watch hrs", "login days", "profiles", "fee",
                                 "avg/day", "churned"], ax=ax)
        ax.grid(False)
        ax.set_title("Correlation heatmap")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
        show(fig)

    # ---- groupby + aggregation
    section("Engagement tiers", "A .groupby() with several aggregations over the filtered customers.", "")
    tiers = (fdf.groupby("engagement_tier")
                .agg(customers=("churned", "size"),
                     churn_rate=("churned", "mean"),
                     avg_watch_hours=("watch_hours", "mean"),
                     avg_days_since_login=("last_login_days", "mean"))
                .reindex([t for t in TIER_ORDER if t in set(fdf["engagement_tier"])]))
    st.dataframe(tiers.style.format({
        "customers": "{:,.0f}", "churn_rate": "{:.1%}",
        "avg_watch_hours": "{:.1f}", "avg_days_since_login": "{:.0f}"}))

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 · MODEL
# ──────────────────────────────────────────────────────────────────────────────
with tab_model:
    section("Can we predict churn?", "A constant baseline versus logistic regression with one-hot encoded plan and payment method.", "")
    yte, p_full, metrics = M["yte"], M["p_full"], M["metrics"]
    coefs = pd.Series(model.coef_[0], index=COLS)

    st.markdown(
        f"""<div class='note'><b>Setup:</b> {M['n_train']:,} training / {M['n_test']:,} test customers
(80/20 stratified split, <code>random_state=42</code>). Target: <code>churned</code> (1 = cancelled).
Features: {', '.join(FEAT_LABEL[c].lower() for c in NUM_FEATS)} plus one-hot encoded
<b>subscription plan</b> and <b>payment method</b>.
<b>Reference levels dropped:</b> plan → <b>{M['refs']['subscription_type']}</b>,
payment → <b>{M['refs']['payment_method']}</b>. <code>monthly_fee</code> is left out because it is an
exact copy of plan (one price per plan), and <code>avg_watch_time_per_day</code> because it is
inconsistent with watch hours and would double count.</div>""",
        unsafe_allow_html=True,
    )

    # ---- model comparison (test set)
    st.markdown("#### Test-set scorecard")
    m1, m2, m3 = st.columns(3)
    names = list(metrics.index)
    for colw, nm in zip((m1, m2, m3), names):
        with colw:
            kpi(nm, f"{metrics.loc[nm, 'Accuracy']:.1%}",
                f"log loss {metrics.loc[nm, 'Log loss']:.3f} · AUC {metrics.loc[nm, 'ROC AUC']:.3f}")

    num_ll = metrics.loc["Logistic · numeric only", "Log loss"]
    full_ll = metrics.loc["Logistic · + one-hot plan & payment", "Log loss"]
    num_acc = metrics.loc["Logistic · numeric only", "Accuracy"]
    full_acc = metrics.loc["Logistic · + one-hot plan & payment", "Accuracy"]
    helped = full_ll < num_ll
    st.markdown(
        f"""<div class='callout'><b>Did the encoded features help?</b>
{'Yes.' if helped else 'Not really.'} Adding plan and payment method moved test log loss from
<b>{num_ll:.3f}</b> to <b>{full_ll:.3f}</b> ({(full_ll - num_ll) / num_ll:+.0%}) and accuracy from
<b>{num_acc:.1%}</b> to <b>{full_acc:.1%}</b>. The constant model, which just guesses the base churn rate
({M['p_const']:.1%}) for everyone, sits at <b>{metrics.iloc[0]['Accuracy']:.1%}</b> accuracy.</div>""",
        unsafe_allow_html=True,
    )

    # ---- interpret an encoded coefficient (interactive widget #6)
    section("Read a coefficient", "Pick a one-hot encoded feature and see what the model says in plain language.", "")
    enc_cols = [c for c in COLS if c not in NUM_FEATS]
    sel = st.selectbox("Encoded feature", enc_cols, format_func=pretty)
    field = next(f for f in ENC_FEATS if sel.startswith(f + "_"))
    level, ref = sel[len(field) + 1:], M["refs"][field]
    odds = float(np.exp(coefs[sel]))

    typical = dict(watch_hours=M["Xtr"]["watch_hours"].median(),
                   last_login_days=M["Xtr"]["last_login_days"].median(),
                   number_of_profiles=M["Xtr"]["number_of_profiles"].median(),
                   age=M["Xtr"]["age"].median(),
                   subscription_type=M["refs"]["subscription_type"],
                   payment_method=M["refs"]["payment_method"])
    p_ref = model.predict_proba(build_rows(**typical))[0, 1]
    p_lvl = model.predict_proba(build_rows(**{**typical, field: level}))[0, 1]
    direction = "higher" if odds > 1 else "lower"
    st.markdown(
        f"""<div class='callout'>Holding everything else constant, a customer with
{FIELD_NAME[field]} = <b>{level}</b> has about <b>{odds:.2f}×</b> the odds of churning of a customer with
<b>{ref}</b> (the reference level), which means odds that are about <b>{abs(odds - 1):.0%} {direction}</b>.
For a typical customer ({typical['watch_hours']:.0f} watch hours, {typical['last_login_days']:.0f} days since
login), predicted churn probability goes from <b>{p_ref:.0%}</b> to <b>{p_lvl:.0%}</b>.</div>""",
        unsafe_allow_html=True,
    )

    # ---- plots: fitted curve, coefficients, calibration, probability separation
    a, b = st.columns(2)
    with a:  # fitted curve over actual data
        grid = np.linspace(0, df["watch_hours"].quantile(0.97), 120)
        fig, ax = plt.subplots(figsize=(6, 4))
        samp = df.sample(1200, random_state=3)
        jitter = np.random.default_rng(0).normal(0, 0.025, len(samp))
        ax.scatter(samp["watch_hours"], samp["churned"] + jitter,
                   s=8, alpha=0.18, color=GREY, linewidth=0)
        for plan, colr in zip(["Basic", "Premium"], [RED, GREEN]):
            rows = build_rows(n=len(grid), **{**typical, "subscription_type": plan})
            rows["watch_hours"] = grid
            ax.plot(grid, model.predict_proba(rows)[:, 1], color=colr, lw=2.6,
                    label=f"{plan} plan (fitted)")
        ax.set_xlim(0, grid.max())
        ax.set_xlabel("Watch hours")
        ax.set_ylabel("Predicted churn probability")
        ax.set_title("Model fit over the actual data")
        ax.legend(loc="center right")
        show(fig)

    with b:  # coefficients (log-odds vs reference level)
        enc = coefs[enc_cols].sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh([pretty(c) for c in enc.index], enc.values,
                color=[RED if v > 0 else GREEN for v in enc.values], height=0.62)
        for i, v in enumerate(enc.values):
            ax.text(v + (0.06 if v > 0 else -0.06), i, f"{np.exp(v):.2f}×",
                    va="center", ha="left" if v > 0 else "right", fontsize=8, color=TXT)
        ax.axvline(0, color=GREY, lw=1)
        lim = np.abs(enc.values).max() * 1.35
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("Log-odds vs. reference level (labels show odds multiplier)")
        ax.set_title("One-hot coefficients")
        ax.grid(axis="y", visible=False)
        show(fig)

    c_, d_ = st.columns(2)
    with c_:  # predicted vs actual (calibration)
        cal = pd.DataFrame({"p": p_full, "y": yte.values})
        cal["bin"] = pd.qcut(cal["p"], 10, duplicates="drop")
        cg = cal.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("y", "mean"))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1, label="perfect")
        ax.plot(cg["pred"], cg["actual"], marker="o", color=RED, lw=2.4, label="model")
        ax.set_xlabel("Predicted churn probability (decile average)")
        ax.set_ylabel("Actual churn rate")
        ax.set_title("Predicted vs. actual (test set)")
        ax.legend(loc="upper left")
        show(fig)

    with d_:  # how well probabilities separate the classes
        fig, ax = plt.subplots(figsize=(6, 4))
        for lab, colr, name in [(0, GREEN, "Stayed"), (1, RED, "Churned")]:
            ax.hist(p_full[yte.values == lab], bins=25, alpha=0.6, color=colr, label=name)
        ax.axvline(0.5, color=TXT, ls=":", lw=1)
        ax.set_xlabel("Predicted churn probability")
        ax.set_ylabel("Test customers")
        ax.set_title("Where the model is confident (and where it isn't)")
        ax.legend()
        show(fig)

    # ---- one evaluation observation, computed from the data
    band = (p_full > 0.35) & (p_full < 0.65)
    correct = (p_full >= 0.5) == (yte.values == 1)
    gap = (cg["pred"] - cg["actual"]).abs().mean() * 100
    st.markdown(
        f"""<div class='callout'><b>Evaluation:</b> predicted probabilities track actual churn rates closely
(average gap {gap:.1f} points across deciles). The model's mistakes cluster in the gray zone:
<b>{band.mean():.0%}</b> of test customers get a probability between 35% and 65% and only
<b>{correct[band].mean():.0%}</b> of those are classified correctly, versus
<b>{correct[~band].mean():.0%}</b> for everyone else.</div>""",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 · PREDICT  (what-if widgets)
# ──────────────────────────────────────────────────────────────────────────────
with tab_predict:
    section("Test a customer", "Build a hypothetical subscriber and watch the predicted churn risk move.", "")
    left, right = st.columns([1, 1.15])
    with left:
        in_plan = st.selectbox("Subscription plan", PLAN_ORDER, index=1, key="in_plan")
        in_pay = st.selectbox("Payment method", sorted(df["payment_method"].unique()),
                              index=0, key="in_pay")
        in_hours = st.slider("Watch hours (per month)", 0.0, 60.0, 8.0, 0.5, key="in_hours")
        in_login = st.slider("Days since last login", 0, 60, 20, key="in_login")
        in_prof = st.slider("Profiles on account", 1, 5, 3, key="in_prof")
        in_age = st.slider("Age", 18, 70, 40, key="in_age")

    row = build_rows(watch_hours=in_hours, last_login_days=in_login,
                     number_of_profiles=in_prof, age=in_age,
                     subscription_type=in_plan, payment_method=in_pay)
    prob = float(model.predict_proba(row)[0, 1])
    level_txt, colr = (("LOW RISK", GREEN) if prob < 0.3
                       else ("MEDIUM RISK", "#f5a524") if prob < 0.6
                       else ("HIGH RISK", RED))

    with right:
        st.markdown(
            f"""<div class='kpi' style='padding:1.6rem 1.8rem'>
<div class='kpi-label'>Predicted probability of churning</div>
<div class='big-prob' style='color:{colr}'>{prob:.0%}</div>
<div class='gauge'><div class='gauge-fill' style='width:{prob * 100:.0f}%;background:{colr}'></div></div>
<div class='kpi-sub'><b style='color:{colr}'>{level_txt}</b> · the average customer in the data churns at {overall_churn:.0%}</div>
</div>""",
            unsafe_allow_html=True,
        )
        # what is pushing this customer up or down vs. the average customer
        contrib = (model.coef_[0] * (row.iloc[0].values - M["Xtr"].mean().values))
        contrib = pd.Series(contrib, index=COLS)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:6].sort_values()
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.barh([pretty(c) for c in top.index], top.values,
                color=[RED if v > 0 else GREEN for v in top.values], height=0.6)
        ax.axvline(0, color=GREY, lw=1)
        ax.set_xlabel("Push on churn risk vs. average customer (log-odds)")
        ax.set_title("What's driving this prediction")
        ax.grid(axis="y", visible=False)
        show(fig)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 · DATA & CLEANING  (Faithfulness & Granularity)
# ──────────────────────────────────────────────────────────────────────────────
with tab_data:
    section("Faithfulness & granularity", "What this data is, and how far to trust it.", "")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(
            f"""<div class='note'><b>What is one row?</b><br>One subscriber account
(<code>customer_id</code> is unique: {audit['dup_ids']} repeated IDs). Each row holds a single snapshot of
that customer's plan, payment method, device, usage and whether they churned.</div>""",
            unsafe_allow_html=True)
    with g2:
        st.markdown(
            f"""<div class='note'><b>What is the scope?</b><br>{audit['raw_rows']:,} customers
across {df['region'].nunique()} regions, ages {df['age'].min()}–{df['age'].max()}. There is <b>no
date column</b>, so we cannot see <i>when</i> people churned, and watch hours are a single aggregate.
Regions, genres and churn (about 50%) are very evenly balanced, which looks more like curated or
simulated data than a raw production log. Treat the findings as patterns in this
dataset, not as claims about Netflix.</div>""",
            unsafe_allow_html=True)
    with g3:
        st.markdown(
            f"""<div class='note'><b>Strange values?</b><br>{audit['dup_rows']} duplicate rows.
<b>{audit['impossible_rows']}</b> rows had more than 24 watch-hours <i>per day</i>, which is physically
impossible, so they were dropped ({audit['impossible_rows'] / audit['raw_rows']:.1%} of data).
<code>avg_watch_time_per_day</code> is usually a small slice of <code>watch_hours</code> but sometimes
equals it, so it is not trusted as a feature. Watch hours are right-skewed with real heavy users
(max {df['watch_hours'].max():.0f}), which are kept.</div>""",
            unsafe_allow_html=True)

    section("Missing values", "Decision rule: under 5% of rows missing → impute; 5% or more → drop the column.", "")
    miss = audit["missing_pct"].rename("% missing").to_frame()
    ca, cb = st.columns([1, 1.4])
    with ca:
        st.dataframe(miss.style.format("{:.2f}%"), height=300)
    with cb:
        st.markdown(
            f"""<div class='note'><b>Result:</b> the highest missing rate in any column is
<b>{miss['% missing'].max():.2f}%</b>, so nothing needed imputing or dropping
(imputed: {audit['imputed'] or 'none'}; dropped columns: {audit['dropped_cols'] or 'none'}).<br><br>
Had a column been missing, the rule above would run automatically: small gaps get a
median/“Unknown” fill because deleting rows would waste data, while large gaps mean the column
is too unreliable to invent. Before imputing, one should also check whether missingness lines up with
<code>churned</code>. If it did, filling values would hide a signal that the missingness itself carries.</div>""",
            unsafe_allow_html=True)

    section("Engineered columns", "Created with .str, pd.cut and .apply.", "")
    st.dataframe(df[["short_id", "watch_hours", "engagement_tier", "age", "age_group",
                     "last_login_days", "login_recency", "churn_label"]].head(10))
    st.caption(f"Rows: {audit['raw_rows']:,} raw → {audit['clean_rows']:,} after cleaning.")
