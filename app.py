#st.set_page_config(page_title="V-EYE", layout="wide")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import time

st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: #00ADB5;
}

.stButton>button {
    background-color: #00ADB5;
    color: white;
    border-radius: 10px;
    padding: 0.5em 1em;
}

.stSelectbox label {
    font-weight: bold;
}

.block-container {
    padding-top: 2rem;
}

.metric-card {
    background-color: #1E1E2F;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)



def demographic_parity(df,target,sensitive):
    groups = df[sensitive].unique()
    rates = {}

    for g in groups:
        subset = df[df[sensitive]==g]
        rate = subset[target].mean()
        rates[g] = rate

    return rates

def balance_data(df, sensitive, target):
    # create groups based on both sensitive + target
    grouped = df.groupby([sensitive, target])

    min_count = grouped.size().min()

    balanced_list = []

    for _, group in grouped:
        sampled = group.sample(min_count, random_state=42)
        balanced_list.append(sampled)

    balanced_df = pd.concat(balanced_list).reset_index(drop=True)

    return balanced_df

def explain_bias(df, target):
    # Drop rows with missing target
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    # Convert categorical to numeric
    #X = pd.get_dummies(X, drop_first=True)

    # limit high-cardinality columns
    for col in X.select_dtypes(include='object').columns:
        if X[col].nunique() < 10:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        else:
            X = X.drop(columns=[col])
    model = RandomForestClassifier()
    model.fit(X, y)

    importance = model.feature_importances_

    feature_importance = sorted(
        zip(X.columns, importance),
        key=lambda x: x[1],
        reverse=True
    )

    return feature_importance

#st.title("FairCare - AI Bias Detector")
#st.write("Upload your dataset to analyze bias")

st.markdown("## V-EYE AI Bias Detector")
st.caption("Detect • Fix • Explain Bias in AI Systems")

# File upload
file = st.file_uploader("Upload CSV file", type=["csv"])

# If file is uploaded
if file is not None:
    df = pd.read_csv(file)
    df = df.ffill()
    df.columns = df.columns.str.strip()
    st.subheader("Dataset Preview")
    st.write("Shape of dataset:", df.shape)
    st.write("Columns:", df.columns.tolist())
    #st.write(df.head())
    st.dataframe(df.head(), use_container_width=True)
    #target = st.selectbox("select target column",df.columns)
    #sensitive = st.selectbox("select sensitive Attribute",df.columns)

    col1, col2 = st.columns(2)

    with col1:
        target = st.selectbox(" Target Column", df.columns)

    with col2:
        sensitive = st.selectbox(" Sensitive Attribute", df.columns)
# Convert target to binary (0/1)
    df[target] = df[target].astype(str).str.strip()

    if df[target].nunique() == 2:
        df[target] = df[target].astype('category').cat.codes
    else:
        st.error("Target must be binary (2 unique values only)")
    

        # Convert target column to numeric
    df[target] = pd.to_numeric(df[target], errors='coerce')

    # values display
    if target and sensitive:
        rates = demographic_parity(df, target, sensitive)

        st.subheader("Bias Analysis")

        for group, rate in rates.items():
          st.write(f"{group}: {rate * 100:.2f}% approval rate")

        # detect bias
        values = list(rates.values())

        if len(values) > 1:
            ratio = min(values) / max(values)

            if ratio < 0.8:
                st.error("Potential bias detected! (Disparate Impact < 0.8)")
            else:
                st.success(" No significant bias detected")
         # chart code
        st.subheader("Bias Visualization")

        fig, ax = plt.subplots()

        percent_values = [v * 100 for v in rates.values()]
        ax.bar(rates.keys(), percent_values)

        ax.set_xlabel("Groups")
        ax.set_ylabel("Approval Rate (%)")
        ax.set_title("Approval Rates by Group")

        st.pyplot(fig)

        st.divider()

        if st.button(" Explain Bias"):
             st.subheader("AI Explanation")

             feature_importance = explain_bias(df, target)

             for feature, score in feature_importance[:5]:
                st.write(f"{feature}: {score:.3f}")

        st.divider()

        if st.button(" Fix Bias"):
            loader_placeholder = st.empty()


            loader_placeholder.empty()  # remove loader

            balanced_df = balance_data(df, sensitive,target)
            new_rates = demographic_parity(balanced_df, target, sensitive)

            st.subheader("After Bias Fix")
            
    
        if new_rates is not None:
            st.write("### Comparison")

            col1, col2 = st.columns(2)

            with col1:
                st.write("Before Fix")
                for group, rate in rates.items():
                    st.write(f"{group}: {rate * 100:.2f}%")

            with col2:
                st.write("After Fix")
                for group, rate in new_rates.items():
                    st.write(f"{group}: {rate * 100:.2f}%")

# visualization code after fix
                st.subheader("After Fix Visualization")
                fig2, ax2 = plt.subplots()
                percent_values_new = [v * 100 for v in new_rates.values()]
                ax2.bar(new_rates.keys(), percent_values_new)
                ax2.set_xlabel("Groups")
                ax2.set_ylabel("Approval Rate (%)")
                ax2.set_title("After Bias Fix")
                st.pyplot(fig2) 
