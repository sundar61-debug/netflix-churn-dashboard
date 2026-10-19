"""
Why Do Netflix Subscribers Leave?
Central question: what separates customers who cancel from customers who stay,
and can a simple model predict who is at risk?
"""
from pathlib import Path  # lets me build file paths that work on any computer

import matplotlib.pyplot as plt  # main plotting library (bar charts, scatter, etc)
import numpy as np  # math/array tools (used for np.full, np.exp, np.linspace)
import pandas as pd  # dataframes: loading, cleaning, grouping the data
import seaborn as sns  # prettier statistical plots built on matplotlib
import streamlit as st  # turns this python file into a web app
from matplotlib.colors import LinearSegmentedColormap  # lets me make a custom color scale for the heatmap
from sklearn.linear_model import LogisticRegression  # the prediction model
from sklearn.metrics import accuracy_score, log_loss  # two ways to score the model
from sklearn.model_selection import train_test_split  # splits data into train and test sets

st.set_page_config(page_title="Netflix Churn", page_icon="🎬", layout="wide",  # browser tab title, tab icon, use full screen width
                   initial_sidebar_state="expanded")  # sidebar starts open

# Netflix aesthetic
RED, GREEN = "#E50914", "#46d369"  # Netflix red = churned, green = stayed (reused in every chart)

# the block below is CSS (styling) injected into the page: dark background, Netflix font, hero box, callout boxes, metric cards
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
""", unsafe_allow_html=True)  # unsafe_allow_html lets streamlit actually use the html/css instead of showing it as text

plt.rcParams.update({  # sets default look for ALL matplotlib charts so they match the dark theme
    "figure.facecolor": "#16161d", "axes.facecolor": "#16161d",  # dark chart background
    "axes.edgecolor": "#2a2a33", "text.color": "#f5f5f1", "axes.labelcolor": "#f5f5f1",  # border + text colors
    "xtick.color": "#b0b0ba", "ytick.color": "#b0b0ba", "grid.color": "#2a2a33",  # axis number + gridline colors
    "axes.spines.top": False, "axes.spines.right": False,  # hide top and right border lines for a cleaner look
})


def callout(text):  # helper so i can make the red-bar highlight boxes with one line
    st.markdown(f"<div class='callout'>{text}</div>", unsafe_allow_html=True)  # wraps the text in the callout css class


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


raw, df, too_many_hours = load_data()  # run the cleaning function once and keep the original, the cleaned data, and the count of impossible rows

# STEP 2: One hot encode- train the models
num_features = ["watch_hours", "last_login_days", "number_of_profiles", "age"]  # number columns the model uses as inputs
cat_features = ["subscription_type", "payment_method"]  # text columns the model uses (need to be turned into numbers first)

X = df[num_features + cat_features]  # X = the input columns (the clues the model learns from)
y = df["churned"]  # y = the answer column (0 = stayed, 1 = churned)
X_train, X_test, y_train, y_test = train_test_split(  # split into a training set and a test set
    X, y, test_size=0.2, random_state=42, stratify=y)  # 20% held out for testing, fixed seed so it's the same every run, stratify keeps the churn % equal in both sets

# one-hot encode; drop_first=True drops Basic (plan) and Credit Card (payment)
X_train = pd.get_dummies(X_train, columns=cat_features, drop_first=True, dtype=float)  # turn each text category into its own 0/1 column (training data)
X_test = pd.get_dummies(X_test, columns=cat_features, drop_first=True, dtype=float)  # do the exact same encoding on the test data
# make sure the test set has exactly the same columns as the train set
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)  # force same column order, fill any missing column with 0

# model 1: constant model (guess the training churn rate for everyone)
base_rate = y_train.mean()  # average churn rate in the training data
const_probs = np.full(len(y_test), base_rate)  # predict that same rate for every test customer (the "no skill" baseline)

# model 2: logistic regression WITHOUT the encoded columns
model_num = LogisticRegression(max_iter=5000).fit(X_train[num_features], y_train)  # train using only the number columns (max_iter gives it enough tries to settle)
num_probs = model_num.predict_proba(X_test[num_features])[:, 1]  # predicted chance of churn for each test customer ([:, 1] = the churn column)

# model 3: logistic regression WITH the encoded columns
model = LogisticRegression(max_iter=5000).fit(X_train, y_train)  # train using number columns + the encoded plan/payment columns
probs = model.predict_proba(X_test)[:, 1]  # predicted churn chance for each test customer from the full model

# Sidebar menu
st.sidebar.markdown("## 🎬 CHURN LAB")  # app title in the sidebar
page = st.sidebar.radio("Go to", ["📊 Explore", "🤖 Model", "🔮 Predict a customer",  # radio buttons to pick a page, the choice gets stored in `page`
                                  "🧹 Data & cleaning"])
st.sidebar.caption("pandas · seaborn · scikit-learn · Streamlit")  # small text listing the tools used

# Page 1: Explore
if page == "📊 Explore":  # only run this block when Explore is selected
    st.markdown("""
    <div class="hero">
      <div class="hero-title">Why do subscribers <span>walk away?</span></div>
      <div class="hero-sub">Explore who cancels and who stays, then see how well a simple model
      can predict it. Use the menu on the left to switch pages.</div>
    </div>
    """, unsafe_allow_html=True)  # big banner at the top using the hero css classes

    # summary numbers side by side
    c1, c2, c3, c4 = st.columns(4)  # make 4 equal columns
    c1.metric("Customers", f"{len(df):,}")  # total number of customers after cleaning (with commas)
    c2.metric("Churn rate", f"{df['churned'].mean():.1%}")  # mean of a 0/1 column = % who churned
    c3.metric("Avg watch hours (stayed)", f"{df.loc[df.churned == 0, 'watch_hours'].mean():.1f}")  # average watch hours for customers who stayed
    c4.metric("Avg watch hours (churned)", f"{df.loc[df.churned == 1, 'watch_hours'].mean():.1f}")  # average watch hours for customers who left

    plan_rate = df.groupby("subscription_type")["churned"].mean()  # churn rate for each plan
    risky = df[df["payment_method"].isin(["Crypto", "Gift Card"])]["churned"].mean()  # churn rate for Crypto + Gift Card payers
    cards = df[df["payment_method"].isin(["Credit Card", "Debit Card"])]["churned"].mean()  # churn rate for card payers
    callout(f"<b>The short answer:</b> engagement and how people pay. Basic plan customers churn at "  # summary box that fills in the real numbers
            f"<b>{plan_rate['Basic']:.0%}</b> vs <b>{plan_rate['Premium']:.0%}</b> on Premium. "
            f"Gift Card and Crypto payers churn at <b>{risky:.0%}</b> vs <b>{cards:.0%}</b> for card payers.")

    left, right = st.columns(2)  # two side by side areas for the first row of charts

    # ---- bar plot (categorical) + selectbox widget
    with left:  # everything indented here goes in the left column
        st.subheader("Churn rate by group")  # section heading
        options = {"Subscription plan": "subscription_type", "Payment method": "payment_method",  # dictionary: nice label shown to user -> real column name
                   "Region": "region", "Device": "device", "Favorite genre": "favorite_genre",
                   "Gender": "gender", "Engagement tier": "engagement_tier",
                   "Login recency": "login_recency"}
        choice = st.selectbox("Break churn down by:", list(options.keys()), index=1)  # dropdown menu, starts on Payment method
        rates = df.groupby(options[choice])["churned"].mean().sort_values(ascending=False) * 100  # churn % for each group, biggest first

        fig, ax = plt.subplots(figsize=(6, 4))  # make a blank chart 6x4 inches
        bars = ax.bar(rates.index, rates.values, color=RED)  # draw one red bar per group
        ax.bar_label(bars, fmt="%.0f%%", color="white")  # write the % on top of each bar
        ax.axhline(df["churned"].mean() * 100, color="gray", linestyle="--")  # dashed line = overall average churn rate for comparison
        ax.set_ylabel("Churn rate (%)")  # y axis label
        ax.set_title(f"Churn rate by {choice.lower()}")  # title changes with the dropdown choice
        plt.xticks(rotation=30)  # tilt the x labels so they don't overlap
        st.pyplot(fig)  # show the chart in the app

    # ---- histogram (quantitative) + radio widget
    with right:  # right column
        st.subheader("Distribution")  # section heading
        hist_options = {"Watch hours": "watch_hours", "Days since last login": "last_login_days",  # label -> column name for the histogram choices
                        "Age": "age"}
        hist_choice = st.radio("Show the distribution of:", list(hist_options.keys()), horizontal=True)  # radio buttons laid out in a row

        fig, ax = plt.subplots(figsize=(6, 4))  # blank chart
        sns.histplot(data=df, x=hist_options[hist_choice], hue="churn_label", bins=30,  # histogram of the chosen column, split by color into stayed vs churned
                     palette={"Stayed": GREEN, "Churned": RED}, element="step", ax=ax)  # green/red colors, "step" draws outlines so the two overlap clearly
        ax.set_title(f"{hist_choice}: stayed vs churned")  # title changes with the radio choice
        st.pyplot(fig)  # show chart

    left2, right2 = st.columns(2)  # two more columns for the second row of charts

    # ---- scatter plot (relationship)
    with left2:  # bottom left
        st.subheader("Watch time vs. last login")  # section heading
        sample = df.sample(1000, random_state=1)  # random 1000 customers (all of them would be too cluttered), fixed seed so it doesn't change
        fig, ax = plt.subplots(figsize=(6, 4))  # blank chart
        sns.scatterplot(data=sample, x="watch_hours", y="last_login_days", hue="churn_label",  # each dot is a customer, colored by whether they churned
                        palette={"Stayed": GREEN, "Churned": RED}, alpha=0.6, ax=ax)  # alpha makes dots see-through so overlaps show
        ax.set_title("Quiet and absent customers churn")  # takeaway as the title
        st.pyplot(fig)  # show chart

    # ---- correlation heatmap
    with right2:  # bottom right
        st.subheader("Correlation heatmap")  # section heading
        nf_colors = LinearSegmentedColormap.from_list("nf", [GREEN, "#16161d", RED])  # custom color scale: green (negative) -> dark (zero) -> red (positive)
        corr = df[["age", "watch_hours", "last_login_days", "number_of_profiles", "churned"]].corr()  # correlation between every pair of these columns
        fig, ax = plt.subplots(figsize=(6, 4))  # blank chart
        sns.heatmap(corr, mask=np.eye(len(corr), dtype=bool), annot=True, fmt=".2f",  # mask hides the diagonal (each column vs itself = 1), annot prints the numbers rounded to 2 decimals
                    cmap=nf_colors, center=0, cbar=False, ax=ax)  # use my colors, make 0 the middle color, hide the color bar
        ax.set_title("What moves with churn?")  # title as a question
        st.pyplot(fig)  # show chart

    # ---- groupby with aggregation
    st.subheader("Engagement tiers")  # section heading
    tiers = df.groupby("engagement_tier").agg(  # group customers by tier and calculate several stats at once
        customers=("churned", "count"),  # how many customers in each tier
        churn_rate=("churned", "mean"),  # churn rate in each tier
        avg_watch_hours=("watch_hours", "mean"),  # average watch hours in each tier
    ).reindex(["Light", "Casual", "Regular", "Binge"])  # put the rows in logical order instead of alphabetical
    st.dataframe(tiers.style.format({"churn_rate": "{:.1%}", "avg_watch_hours": "{:.1f}"}))  # show the table, churn as a % and hours to 1 decimal

# Page 2: Model
elif page == "🤖 Model":  # only run this block when Model is selected
    st.markdown("<div class='page-title'>CAN WE PREDICT CHURN?</div>", unsafe_allow_html=True)  # red page title
    callout(f"<b>Setup:</b> 80% of customers train the model ({len(X_train):,}) and 20% test it "  # box explaining the train/test split with real counts
            f"({len(X_test):,}). The split is random but fixed (random_state=42) and keeps the same "
            f"churn rate in both. <b>Dropped reference levels:</b> plan = <b>Basic</b>, "
            f"payment method = <b>Credit Card</b>.")

    # scoreboard
    acc_const = accuracy_score(y_test, const_probs >= 0.5)  # accuracy of the constant model (turn probability into a 0/1 guess at the 50% cutoff)
    acc_num = accuracy_score(y_test, num_probs >= 0.5)  # accuracy of the model without encoded columns
    acc_full = accuracy_score(y_test, probs >= 0.5)  # accuracy of the full model
    ll_const, ll_num, ll_full = log_loss(y_test, const_probs), log_loss(y_test, num_probs), log_loss(y_test, probs)  # log loss for all 3 (punishes confident wrong guesses, lower = better)

    st.subheader("Test-set scoreboard")  # section heading
    m1, m2, m3 = st.columns(3)  # 3 columns, one per model
    m1.metric("Constant model", f"{acc_const:.1%}", f"log loss {ll_const:.3f}", delta_color="off")  # accuracy big, log loss underneath (delta_color off = no green/red arrow color)
    m2.metric("Logistic (no encoded features)", f"{acc_num:.1%}", f"log loss {ll_num:.3f}", delta_color="off")  # same for model 2
    m3.metric("Logistic + one-hot plan & payment", f"{acc_full:.1%}", f"log loss {ll_full:.3f}", delta_color="off")  # same for model 3

    verdict = "Yes, it helped." if ll_full < ll_num else "Not really."  # if the full model has lower log loss, the encoded columns helped
    callout(f"<b>Did the encoded feature help?</b> {verdict} Test log loss went from "  # box that states the verdict with the real before/after numbers
            f"<b>{ll_num:.3f}</b> to <b>{ll_full:.3f}</b> (lower is better) and accuracy from "
            f"<b>{acc_num:.1%}</b> to <b>{acc_full:.1%}</b>.")

    # coefficient explainer
    st.subheader("Read a coefficient")  # section heading
    coefs = pd.Series(model.coef_[0], index=X_train.columns)  # the model's learned weights, labeled with the column each belongs to
    encoded_cols = [c for c in X_train.columns if c not in num_features]  # just the one-hot columns (plan and payment ones)
    pick = st.selectbox("Pick an encoded feature:", encoded_cols)  # dropdown so the audience chooses which one to explain
    odds = np.exp(coefs[pick])  # exp turns the coefficient into an odds multiplier
    if pick.startswith("subscription_type_"):  # if the picked column is a plan...
        group, level, reference = "plan", pick.replace("subscription_type_", ""), "Basic"  # ...name the group, the level (e.g. Premium), and the dropped reference (Basic)
    else:  # otherwise it's a payment method
        group, level, reference = "payment method", pick.replace("payment_method_", ""), "Credit Card"  # same idea, reference is Credit Card
    more_less = "higher" if odds > 1 else "lower"  # above 1 means more likely to churn, below 1 means less likely
    callout(f"Holding everything else constant, a customer whose {group} is <b>{level}</b> has about "  # plain-english sentence explaining the coefficient
            f"<b>{odds:.2f}x</b> the odds of churning compared with <b>{reference}</b> (the reference "
            f"level). That is about <b>{abs(odds - 1):.0%} {more_less}</b> odds.")

    table = pd.DataFrame({"coefficient": coefs.round(3), "odds multiplier": np.exp(coefs).round(2)})  # table with every coefficient and its odds multiplier
    st.dataframe(table)  # show it

    left, right = st.columns(2)  # two columns for the charts

    # fitted curve on top of the real data
    with left:  # left column
        st.subheader("Model vs. actual data")  # section heading
        hours = np.linspace(0, 40, 100)  # 100 evenly spaced watch-hour values from 0 to 40 for drawing a smooth curve
        fig, ax = plt.subplots(figsize=(6, 4))  # blank chart
        sample = df.sample(1000, random_state=2)  # random 1000 real customers
        ax.scatter(sample["watch_hours"], sample["churned"] + np.random.normal(0, 0.02, len(sample)),  # plot real data; tiny random noise (jitter) keeps the 0s and 1s from stacking on one line
                   s=8, alpha=0.2, color="gray")  # small, see-through gray dots
        for plan, color in [("Basic", RED), ("Premium", GREEN)]:  # draw one prediction curve per plan
            grid = pd.DataFrame(0.0, index=range(100), columns=X_train.columns)  # fake customers: 100 rows, all zeros, same columns the model expects
            grid["watch_hours"] = hours  # only watch hours changes across the rows
            grid["last_login_days"] = df["last_login_days"].median()  # hold login days at the typical (median) value
            grid["number_of_profiles"] = df["number_of_profiles"].median()  # hold profiles at the median
            grid["age"] = df["age"].median()  # hold age at the median
            if plan != "Basic":  # Basic is the dropped reference level so it stays all zeros
                grid[f"subscription_type_{plan}"] = 1.0  # flip the matching plan column to 1
            ax.plot(hours, model.predict_proba(grid)[:, 1], color=color, linewidth=3, label=f"{plan} (predicted)")  # draw the model's predicted churn chance across watch hours
        ax.set_xlabel("Watch hours")  # x axis label
        ax.set_ylabel("Churn probability")  # y axis label
        ax.legend()  # show which line is which plan
        st.pyplot(fig)  # show chart

    # predicted probabilities split by what really happened
    with right:  # right column
        st.subheader("Predicted vs. actual outcome")  # section heading
        fig, ax = plt.subplots(figsize=(6, 4))  # blank chart
        ax.hist(probs[y_test.values == 0], bins=25, alpha=0.7, color=GREEN, label="Actually stayed")  # histogram of predictions for customers who really stayed
        ax.hist(probs[y_test.values == 1], bins=25, alpha=0.7, color=RED, label="Actually churned")  # histogram of predictions for customers who really left
        ax.set_xlabel("Predicted churn probability")  # x axis label
        ax.set_ylabel("Customers")  # y axis label
        ax.legend()  # show which color is which
        st.pyplot(fig)  # show chart

    # one evaluation observation
    unsure = (probs > 0.35) & (probs < 0.65)  # True for customers where the model is close to a coin flip
    right_answer = (probs >= 0.5) == (y_test.values == 1)  # True when the model's guess matches what really happened
    callout(f"<b>Evaluation:</b> the model is right about <b>{acc_full:.0%}</b> of the time on customers "  # box with the overall accuracy plus accuracy on the "unsure" group
            f"it has never seen. It struggles most when it is unsure: for the <b>{unsure.mean():.0%}</b> of "
            f"customers with a predicted probability between 35% and 65%, it is only right "
            f"<b>{right_answer[unsure].mean():.0%}</b> of the time.")

# Page 3: Predict a customer
elif page == "🔮 Predict a customer":  # only run this block when Predict is selected
    st.markdown("<div class='page-title'>TEST A CUSTOMER</div>", unsafe_allow_html=True)  # red page title
    st.write("Build a made-up subscriber and see how likely they are to cancel.")  # instructions for the audience

    left, right = st.columns(2)  # inputs on the left, result on the right
    with left:  # left column holds all the input widgets
        plan = st.selectbox("Subscription plan", ["Basic", "Standard", "Premium"], index=1)  # dropdown, starts on Standard
        payment = st.selectbox("Payment method", sorted(df["payment_method"].unique()))  # dropdown built from the payment methods in the data, alphabetical
        hours = st.slider("Watch hours per month", 0.0, 60.0, 8.0, 0.5)  # slider: min 0, max 60, starts at 8, moves in 0.5 steps
        login = st.slider("Days since last login", 0, 60, 20)  # slider: 0 to 60, starts at 20
        profiles = st.slider("Profiles on account", 1, 5, 3)  # slider: 1 to 5, starts at 3
        age = st.slider("Age", 18, 70, 40)  # slider: 18 to 70, starts at 40

    # build one row with the same columns the model was trained on
    row = pd.DataFrame(0.0, index=[0], columns=X_train.columns)  # one fake customer, every column starts at 0
    row["watch_hours"], row["last_login_days"] = hours, login  # fill in the slider values for watch hours and login days
    row["number_of_profiles"], row["age"] = profiles, age  # fill in profiles and age
    if f"subscription_type_{plan}" in row.columns:  # Basic has no column (it was dropped), so only set it if the column exists
        row[f"subscription_type_{plan}"] = 1.0  # turn on the chosen plan
    if f"payment_method_{payment}" in row.columns:  # same check for payment (Credit Card was dropped)
        row[f"payment_method_{payment}"] = 1.0  # turn on the chosen payment method
    prob = model.predict_proba(row)[0, 1]  # the model's predicted chance this customer churns

    with right:  # right column shows the result
        st.markdown("#### Chance of churning")  # small heading
        st.markdown(f"<div class='hero-title' style='font-size:5rem'>{prob:.0%}</div>",  # big percentage number using the Netflix font
                    unsafe_allow_html=True)
        st.progress(float(prob))  # progress bar filled to the churn probability
        if prob < 0.3:  # under 30% = low risk
            st.success("🟢 Low risk")  # green box
        elif prob < 0.6:  # 30% to 60% = medium risk
            st.warning("🟡 Medium risk")  # yellow box
        else:  # 60% or more = high risk
            st.error("🔴 High risk")  # red box
        st.caption(f"The average customer in the data churns {df['churned'].mean():.0%} of the time.")  # small text giving the baseline to compare against

#Page 4: Data and cleaning
elif page == "🧹 Data & cleaning":  # only run this block when Data & cleaning is selected
    st.markdown("<div class='page-title'>FAITHFULNESS & GRANULARITY</div>", unsafe_allow_html=True)  # red page title

    c1, c2, c3 = st.columns(3)  # three columns for three boxes
    with c1:  # box 1: granularity
        callout(f"<b>What is one row?</b><br>One subscriber account. Each row is a snapshot of that "  # explains what one row means and counts repeated IDs in the raw data
                f"customer's plan, payment method, device, usage and whether they churned. "
                f"{raw['customer_id'].duplicated().sum()} customer IDs repeat.")
    with c2:  # box 2: scope
        callout(f"<b>What is the scope?</b><br>{len(raw):,} customers in {df['region'].nunique()} regions, "  # explains who/what is covered and warns that the data looks simulated
                f"ages {df['age'].min()} to {df['age'].max()}. There is no date column, so we can't see "
                f"<i>when</i> people left. The data is very evenly balanced (about 50% churn), so it "
                f"looks simulated. Treat the results as patterns in this dataset, not facts about Netflix.")
    with c3:  # box 3: strange values
        callout(f"<b>Strange values?</b><br>{raw.duplicated().sum()} duplicate rows. "  # counts duplicate rows and the impossible >24 hour rows, and explains which columns were left out of the model
                f"{too_many_hours} rows claimed more than 24 hours watched per day, which is impossible, "
                f"so they were dropped. <code>avg_watch_time_per_day</code> is inconsistent with "
                f"<code>watch_hours</code> and <code>monthly_fee</code> just repeats the plan, so neither "
                f"is used in the model.")

    st.subheader("Missing values")  # section heading
    missing = (raw.isna().mean() * 100).round(2).rename("% missing")  # % missing in each column of the raw data, rounded to 2 decimals
    st.dataframe(missing)  # show it as a table
    callout(f"The most missing in any column is <b>{missing.max():.2f}%</b>, so nothing was dropped or "  # box that gives the worst column and explains the 5% rule
            f"filled in. The rule in the code: under 5% missing means fill it in (median for numbers), "
            f"5% or more means drop the column, because too much of it would be made up. Before filling "
            f"anything, you would also check whether the missing rows churn more or less than the rest.")

    st.subheader("New columns I created")  # section heading
    st.write("`short_id` (with .str), `engagement_tier` (with pd.cut), `login_recency` (with .apply)")  # lists the new columns and which pandas tool made each
    st.dataframe(df[["short_id", "watch_hours", "engagement_tier", "last_login_days",  # show the first 10 rows of the new columns next to the columns they came from
                     "login_recency", "churn_label"]].head(10))
    st.caption(f"{len(raw):,} rows before cleaning, {len(df):,} after.")  # small text showing how many rows the cleaning removed
