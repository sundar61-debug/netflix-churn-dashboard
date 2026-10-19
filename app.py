"""
Why Do Netflix Subscribers Leave?
Central question: what separates customers who cancel from customers who stay,
and can a simple model predict who is at risk?
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Netflix Churn", page_icon="🎬", layout="wide",
                   initial_sidebar_state="expanded")

# Netflix aesthetic
RED, GREEN = "#E50914", "#46d369"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');
.stApp { background-color: #0b0b0f; color: #f5f5f1; }
section[data-testid="stSidebar"] { background-color: #12121a; }
.hero { background: linear-gradient(135deg, #4a0b10, #0b0b0f);
        border: 1px solid #2a2a33; border-radius: 18px; padding: 2rem 2.4rem; margin-bottom: 1rem; }
.hero-title { font-family: 'Bebas Neue', sans-serif; font-size: 3.6rem; line-height: 1; color: #f5f5f1; }
.hero-title span { color: #E50914; }
.hero-sub { color: #b0b0ba; margin-top: .5rem; }
.page-title { font-family: 'Bebas Neue', sans-serif; font-size: 2.4rem; color: #E50914; }
.callout { border-left: 4px solid #E50914; background: #16161d; border-radius: 8px;
           padding: .9rem 1.2rem; margin: .8rem 0; }
[data-testid="stMetric"] { background: #16161d; border: 1px solid #2a2a33;
                           border-radius: 12px; padding: .9rem 1rem; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#16161d", "axes.facecolor": "#16161d",
    "axes.edgecolor": "#2a2a33", "text.color": "#f5f5f1", "axes.labelcolor": "#f5f5f1",
    "xtick.color": "#b0b0ba", "ytick.color": "#b0b0ba", "grid.color": "#2a2a33",
    "axes.spines.top": False, "axes.spines.right": False,
})


def callout(text):
    st.markdown(f"<div class='callout'>{text}</div>", unsafe_allow_html=True)


# STEP 1: Load and clean data
@st.cache_data #Make's streamlit remember the data functions (doesn't reload everytime)
def load_data():
    raw = pd.read_csv(Path(__file__).parent / "data" / "netflix_customer_churn.csv") # original to use as comparison
    df = raw.copy() # Data i will mutate

    # remove stray spaces in text columns
    text_cols = ["subscription_type", "payment_method", "region", "device",
                 "favorite_genre", "gender"]
    for c in text_cols:
        df[c] = df[c].str.strip() # remove additional spaces in data

    # missing values: under 5% missing -> fill in, 5% or more -> drop the column
    for c in df.columns:
        pct_missing = df[c].isna().mean() * 100 #find number of total columns with missing values
        if 0 < pct_missing < 5: # if that amount is less than 5 percent
            if pd.api.types.is_numeric_dtype(df[c]): #if it is a numeric value
                df[c] = df[c].fillna(df[c].median()) # fill it with the median value
            else:
                df[c] = df[c].fillna("Unknown") # str is unknown instead of median
        elif pct_missing >= 5: # if missing values are more tha 5% of total data
            df = df.drop(columns=c) #drop the na columns

    # remove duplicates and impossible values (more than 24 hours watched in a day)
    df = df.drop_duplicates(subset="customer_id") # drop duplicate customers using id
    too_many_hours = int((df["avg_watch_time_per_day"] > 24).sum()) # number of customers with >24hr in a day
    df = df[df["avg_watch_time_per_day"] <= 24].reset_index(drop=True) #keep the rest and reset index after dropping columns

    # new columns
    df["short_id"] = df["customer_id"].str[:8] # idk the purpose j to show the str bs
    df["churn_label"] = df["churned"].map({0: "Stayed", 1: "Churned"}) # redo churn so it has labels for key 
    df["engagement_tier"] = pd.cut(df["watch_hours"], bins=[-1, 3, 8, 16, 1000], #column to categorize watch hours so i can visualize
                                   labels=["Light", "Casual", "Regular", "Binge"]).astype(str)
    df["login_recency"] = df["last_login_days"].apply( # applying function to clasify last login status
        lambda d: "This week" if d <= 7 else ("This month" if d <= 30 else "Gone quiet"))
    return raw, df, too_many_hours


raw, df, too_many_hours = load_data() 

# STEP 2: One hot encode- train the models
num_features = ["watch_hours", "last_login_days", "number_of_profiles", "age"]
cat_features = ["subscription_type", "payment_method"]

X = df[num_features + cat_features]
y = df["churned"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# one-hot encode; drop_first=True drops Basic (plan) and Credit Card (payment)
X_train = pd.get_dummies(X_train, columns=cat_features, drop_first=True, dtype=float)
X_test = pd.get_dummies(X_test, columns=cat_features, drop_first=True, dtype=float)
# make sure the test set has exactly the same columns as the train set
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# model 1: constant model (guess the training churn rate for everyone)
base_rate = y_train.mean()
const_probs = np.full(len(y_test), base_rate)

# model 2: logistic regression WITHOUT the encoded columns
model_num = LogisticRegression(max_iter=5000).fit(X_train[num_features], y_train)
num_probs = model_num.predict_proba(X_test[num_features])[:, 1]

# model 3: logistic regression WITH the encoded columns
model = LogisticRegression(max_iter=5000).fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]

# Sidebar menu
st.sidebar.markdown("## 🎬 CHURN LAB")
page = st.sidebar.radio("Go to", ["📊 Explore", "🤖 Model", "🔮 Predict a customer",
                                  "🧹 Data & cleaning"])
st.sidebar.caption("pandas · seaborn · scikit-learn · Streamlit")

# Page 1: Explore
if page == "📊 Explore":
    st.markdown("""
    <div class="hero">
      <div class="hero-title">Why do subscribers <span>walk away?</span></div>
      <div class="hero-sub">Explore who cancels and who stays, then see how well a simple model
      can predict it. Use the menu on the left to switch pages.</div>
    </div>
    """, unsafe_allow_html=True)

    # summary numbers side by side
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{df['churned'].mean():.1%}")
    c3.metric("Avg watch hours (stayed)", f"{df.loc[df.churned == 0, 'watch_hours'].mean():.1f}")
    c4.metric("Avg watch hours (churned)", f"{df.loc[df.churned == 1, 'watch_hours'].mean():.1f}")

    plan_rate = df.groupby("subscription_type")["churned"].mean()
    risky = df[df["payment_method"].isin(["Crypto", "Gift Card"])]["churned"].mean()
    cards = df[df["payment_method"].isin(["Credit Card", "Debit Card"])]["churned"].mean()
    callout(f"<b>The short answer:</b> engagement and how people pay. Basic plan customers churn at "
            f"<b>{plan_rate['Basic']:.0%}</b> vs <b>{plan_rate['Premium']:.0%}</b> on Premium. "
            f"Gift Card and Crypto payers churn at <b>{risky:.0%}</b> vs <b>{cards:.0%}</b> for card payers.")

    left, right = st.columns(2)

    # ---- bar plot (categorical) + selectbox widget
    with left:
        st.subheader("Churn rate by group")
        options = {"Subscription plan": "subscription_type", "Payment method": "payment_method",
                   "Region": "region", "Device": "device", "Favorite genre": "favorite_genre",
                   "Gender": "gender", "Engagement tier": "engagement_tier",
                   "Login recency": "login_recency"}
        choice = st.selectbox("Break churn down by:", list(options.keys()), index=1)
        rates = df.groupby(options[choice])["churned"].mean().sort_values(ascending=False) * 100

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(rates.index, rates.values, color=RED)
        ax.bar_label(bars, fmt="%.0f%%", color="white")
        ax.axhline(df["churned"].mean() * 100, color="gray", linestyle="--")
        ax.set_ylabel("Churn rate (%)")
        ax.set_title(f"Churn rate by {choice.lower()}")
        plt.xticks(rotation=30)
        st.pyplot(fig)

    # ---- histogram (quantitative) + radio widget
    with right:
        st.subheader("Distribution")
        hist_options = {"Watch hours": "watch_hours", "Days since last login": "last_login_days",
                        "Age": "age"}
        hist_choice = st.radio("Show the distribution of:", list(hist_options.keys()), horizontal=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df, x=hist_options[hist_choice], hue="churn_label", bins=30,
                     palette={"Stayed": GREEN, "Churned": RED}, element="step", ax=ax)
        ax.set_title(f"{hist_choice}: stayed vs churned")
        st.pyplot(fig)

    left2, right2 = st.columns(2)

    # ---- scatter plot (relationship)
    with left2:
        st.subheader("Watch time vs. last login")
        sample = df.sample(1000, random_state=1)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=sample, x="watch_hours", y="last_login_days", hue="churn_label",
                        palette={"Stayed": GREEN, "Churned": RED}, alpha=0.6, ax=ax)
        ax.set_title("Quiet and absent customers churn")
        st.pyplot(fig)

    # ---- correlation heatmap
    with right2:
        st.subheader("Correlation heatmap")
        nf_colors = LinearSegmentedColormap.from_list("nf", [GREEN, "#16161d", RED])
        corr = df[["age", "watch_hours", "last_login_days", "number_of_profiles", "churned"]].corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, mask=np.eye(len(corr), dtype=bool), annot=True, fmt=".2f",
                    cmap=nf_colors, center=0, cbar=False, ax=ax)
        ax.set_title("What moves with churn?")
        st.pyplot(fig)

    # ---- groupby with aggregation
    st.subheader("Engagement tiers")
    tiers = df.groupby("engagement_tier").agg(
        customers=("churned", "count"),
        churn_rate=("churned", "mean"),
        avg_watch_hours=("watch_hours", "mean"),
    ).reindex(["Light", "Casual", "Regular", "Binge"])
    st.dataframe(tiers.style.format({"churn_rate": "{:.1%}", "avg_watch_hours": "{:.1f}"}))

# Page 2: Model
elif page == "🤖 Model":
    st.markdown("<div class='page-title'>CAN WE PREDICT CHURN?</div>", unsafe_allow_html=True)
    callout(f"<b>Setup:</b> 80% of customers train the model ({len(X_train):,}) and 20% test it "
            f"({len(X_test):,}). The split is random but fixed (random_state=42) and keeps the same "
            f"churn rate in both. <b>Dropped reference levels:</b> plan = <b>Basic</b>, "
            f"payment method = <b>Credit Card</b>.")

    # scoreboard
    acc_const = accuracy_score(y_test, const_probs >= 0.5)
    acc_num = accuracy_score(y_test, num_probs >= 0.5)
    acc_full = accuracy_score(y_test, probs >= 0.5)
    ll_const, ll_num, ll_full = log_loss(y_test, const_probs), log_loss(y_test, num_probs), log_loss(y_test, probs)

    st.subheader("Test-set scoreboard")
    m1, m2, m3 = st.columns(3)
    m1.metric("Constant model", f"{acc_const:.1%}", f"log loss {ll_const:.3f}", delta_color="off")
    m2.metric("Logistic (no encoded features)", f"{acc_num:.1%}", f"log loss {ll_num:.3f}", delta_color="off")
    m3.metric("Logistic + one-hot plan & payment", f"{acc_full:.1%}", f"log loss {ll_full:.3f}", delta_color="off")

    verdict = "Yes, it helped." if ll_full < ll_num else "Not really."
    callout(f"<b>Did the encoded feature help?</b> {verdict} Test log loss went from "
            f"<b>{ll_num:.3f}</b> to <b>{ll_full:.3f}</b> (lower is better) and accuracy from "
            f"<b>{acc_num:.1%}</b> to <b>{acc_full:.1%}</b>.")

    # coefficient explainer
    st.subheader("Read a coefficient")
    coefs = pd.Series(model.coef_[0], index=X_train.columns)
    encoded_cols = [c for c in X_train.columns if c not in num_features]
    pick = st.selectbox("Pick an encoded feature:", encoded_cols)
    odds = np.exp(coefs[pick])
    if pick.startswith("subscription_type_"):
        group, level, reference = "plan", pick.replace("subscription_type_", ""), "Basic"
    else:
        group, level, reference = "payment method", pick.replace("payment_method_", ""), "Credit Card"
    more_less = "higher" if odds > 1 else "lower"
    callout(f"Holding everything else constant, a customer whose {group} is <b>{level}</b> has about "
            f"<b>{odds:.2f}x</b> the odds of churning compared with <b>{reference}</b> (the reference "
            f"level). That is about <b>{abs(odds - 1):.0%} {more_less}</b> odds.")

    table = pd.DataFrame({"coefficient": coefs.round(3), "odds multiplier": np.exp(coefs).round(2)})
    st.dataframe(table)

    left, right = st.columns(2)

    # fitted curve on top of the real data
    with left:
        st.subheader("Model vs. actual data")
        hours = np.linspace(0, 40, 100)
        fig, ax = plt.subplots(figsize=(6, 4))
        sample = df.sample(1000, random_state=2)
        ax.scatter(sample["watch_hours"], sample["churned"] + np.random.normal(0, 0.02, len(sample)),
                   s=8, alpha=0.2, color="gray")
        for plan, color in [("Basic", RED), ("Premium", GREEN)]:
            grid = pd.DataFrame(0.0, index=range(100), columns=X_train.columns)
            grid["watch_hours"] = hours
            grid["last_login_days"] = df["last_login_days"].median()
            grid["number_of_profiles"] = df["number_of_profiles"].median()
            grid["age"] = df["age"].median()
            if plan != "Basic":
                grid[f"subscription_type_{plan}"] = 1.0
            ax.plot(hours, model.predict_proba(grid)[:, 1], color=color, linewidth=3, label=f"{plan} (predicted)")
        ax.set_xlabel("Watch hours")
        ax.set_ylabel("Churn probability")
        ax.legend()
        st.pyplot(fig)

    # predicted probabilities split by what really happened
    with right:
        st.subheader("Predicted vs. actual outcome")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(probs[y_test.values == 0], bins=25, alpha=0.7, color=GREEN, label="Actually stayed")
        ax.hist(probs[y_test.values == 1], bins=25, alpha=0.7, color=RED, label="Actually churned")
        ax.set_xlabel("Predicted churn probability")
        ax.set_ylabel("Customers")
        ax.legend()
        st.pyplot(fig)

    # one evaluation observation
    unsure = (probs > 0.35) & (probs < 0.65)
    right_answer = (probs >= 0.5) == (y_test.values == 1)
    callout(f"<b>Evaluation:</b> the model is right about <b>{acc_full:.0%}</b> of the time on customers "
            f"it has never seen. It struggles most when it is unsure: for the <b>{unsure.mean():.0%}</b> of "
            f"customers with a predicted probability between 35% and 65%, it is only right "
            f"<b>{right_answer[unsure].mean():.0%}</b> of the time.")

# Page 3: Predict a customer
elif page == "🔮 Predict a customer":
    st.markdown("<div class='page-title'>TEST A CUSTOMER</div>", unsafe_allow_html=True)
    st.write("Build a made-up subscriber and see how likely they are to cancel.")

    left, right = st.columns(2)
    with left:
        plan = st.selectbox("Subscription plan", ["Basic", "Standard", "Premium"], index=1)
        payment = st.selectbox("Payment method", sorted(df["payment_method"].unique()))
        hours = st.slider("Watch hours per month", 0.0, 60.0, 8.0, 0.5)
        login = st.slider("Days since last login", 0, 60, 20)
        profiles = st.slider("Profiles on account", 1, 5, 3)
        age = st.slider("Age", 18, 70, 40)

    # build one row with the same columns the model was trained on
    row = pd.DataFrame(0.0, index=[0], columns=X_train.columns)
    row["watch_hours"], row["last_login_days"] = hours, login
    row["number_of_profiles"], row["age"] = profiles, age
    if f"subscription_type_{plan}" in row.columns:
        row[f"subscription_type_{plan}"] = 1.0
    if f"payment_method_{payment}" in row.columns:
        row[f"payment_method_{payment}"] = 1.0
    prob = model.predict_proba(row)[0, 1]

    with right:
        st.markdown("#### Chance of churning")
        st.markdown(f"<div class='hero-title' style='font-size:5rem'>{prob:.0%}</div>",
                    unsafe_allow_html=True)
        st.progress(float(prob))
        if prob < 0.3:
            st.success("🟢 Low risk")
        elif prob < 0.6:
            st.warning("🟡 Medium risk")
        else:
            st.error("🔴 High risk")
        st.caption(f"The average customer in the data churns {df['churned'].mean():.0%} of the time.")

#Page 4: Data and cleaning
elif page == "🧹 Data & cleaning":
    st.markdown("<div class='page-title'>FAITHFULNESS & GRANULARITY</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        callout(f"<b>What is one row?</b><br>One subscriber account. Each row is a snapshot of that "
                f"customer's plan, payment method, device, usage and whether they churned. "
                f"{raw['customer_id'].duplicated().sum()} customer IDs repeat.")
    with c2:
        callout(f"<b>What is the scope?</b><br>{len(raw):,} customers in {df['region'].nunique()} regions, "
                f"ages {df['age'].min()} to {df['age'].max()}. There is no date column, so we can't see "
                f"<i>when</i> people left. The data is very evenly balanced (about 50% churn), so it "
                f"looks simulated. Treat the results as patterns in this dataset, not facts about Netflix.")
    with c3:
        callout(f"<b>Strange values?</b><br>{raw.duplicated().sum()} duplicate rows. "
                f"{too_many_hours} rows claimed more than 24 hours watched per day, which is impossible, "
                f"so they were dropped. <code>avg_watch_time_per_day</code> is inconsistent with "
                f"<code>watch_hours</code> and <code>monthly_fee</code> just repeats the plan, so neither "
                f"is used in the model.")

    st.subheader("Missing values")
    missing = (raw.isna().mean() * 100).round(2).rename("% missing")
    st.dataframe(missing)
    callout(f"The most missing in any column is <b>{missing.max():.2f}%</b>, so nothing was dropped or "
            f"filled in. The rule in the code: under 5% missing means fill it in (median for numbers), "
            f"5% or more means drop the column, because too much of it would be made up. Before filling "
            f"anything, you would also check whether the missing rows churn more or less than the rest.")

    st.subheader("New columns I created")
    st.write("`short_id` (with .str), `engagement_tier` (with pd.cut), `login_recency` (with .apply)")
    st.dataframe(df[["short_id", "watch_hours", "engagement_tier", "last_login_days",
                     "login_recency", "churn_label"]].head(10))
    st.caption(f"{len(raw):,} rows before cleaning, {len(df):,} after.")
