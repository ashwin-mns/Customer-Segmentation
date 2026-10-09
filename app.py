
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Behaviour Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_PATH = (
    BASE_DIR / "deployment_artifacts" / "dbscan_bundle.joblib"
)

SPENDING_COLS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

PURCHASE_COLS = [
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases"
]

CAMPAIGN_COLS = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "Response"
]

ENGINEERED_COLS = [
    "Age",
    "Total_Children",
    "Total_Spending",
    "Total_Purchases",
    "Total_Campaign_Acceptance"
]


# =========================================================
# 2. LOAD TRAINED ARTIFACTS
# =========================================================

@st.cache_resource
def load_artifacts(path, modified_time):
    return joblib.load(path)


st.title("📊 Customer Behaviour & Segmentation")

st.write(
    "Understand customer spending, purchasing habits, "
    "shopping channels and marketing engagement."
)

if not ARTIFACT_PATH.exists():
    st.error(
        f"Deployment bundle not found: {ARTIFACT_PATH}"
    )
    st.stop()

try:
    artifacts = load_artifacts(
        str(ARTIFACT_PATH),
        ARTIFACT_PATH.stat().st_mtime
    )
except Exception as exc:
    st.error(f"Unable to load model artifacts: {exc}")
    st.stop()


# =========================================================
# 3. PREPROCESS RAW CUSTOMER DATA
# =========================================================

def preprocess_customers(raw, artifacts):

    data = raw.copy()
    data.columns = data.columns.astype(str).str.strip()

    if data.columns.duplicated().any():
        raise ValueError(
            "Duplicate column names found in the uploaded file."
        )

    feature_columns = artifacts["feature_columns"]

    # Separate original model inputs from engineered features.
    original_features = [
        col for col in feature_columns
        if col not in ENGINEERED_COLS
    ]

    # IMPORTANT:
    # Engineered columns are NOT required in the uploaded file.
    # They will be calculated below.
    required_columns = set(original_features)

    # Raw columns needed for feature engineering.
    required_columns.update([
        "Year_Birth",
        "Kidhome",
        "Teenhome",
        *SPENDING_COLS,
        *PURCHASE_COLS,
        *CAMPAIGN_COLS
    ])

    missing_columns = sorted(
        required_columns - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required original dataset columns: "
            + ", ".join(missing_columns)
        )

    # -----------------------------------------------------
    # Numeric conversion
    # -----------------------------------------------------

    categorical_columns = {
        "Education",
        "Marital_Status"
    }

    numeric_columns = [
        col for col in required_columns
        if col not in categorical_columns
    ]

    for col in numeric_columns:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    # Use the income mean saved from the training dataset.
    if "Income" in data.columns:
        data["Income"] = data["Income"].fillna(
            artifacts["income_mean"]
        )

    # -----------------------------------------------------
    # Encode categorical columns using training mappings
    # -----------------------------------------------------

    category_mappings = {
        "Education": artifacts["education_categories"],
        "Marital_Status": artifacts["marital_categories"]
    }

    valid_rows = pd.Series(
        True,
        index=data.index
    )

    encoded_columns = {}

    for col, categories in category_mappings.items():

        if col not in data.columns:
            raise ValueError(
                f"Missing categorical column: {col}"
            )

        categories = [
            str(value) for value in categories
        ]

        mapping = {
            category: index
            for index, category in enumerate(categories)
        }

        values = (
            data[col]
            .astype("string")
            .str.strip()
        )

        encoded = values.map(mapping)

        encoded_columns[col] = encoded

        # Exclude missing or unseen categories consistently.
        valid_rows &= encoded.notna()

    valid_index = data.index[valid_rows]

    excluded_count = len(data) - len(valid_index)

    data = data.loc[valid_index].copy()

    if data.empty:
        raise ValueError(
            "No valid customer records remain after category "
            "validation. Check Education and Marital_Status."
        )

    # Assign encoded values only to retained rows.
    for col in category_mappings:
        data[col] = (
            encoded_columns[col]
            .loc[data.index]
            .astype(int)
        )

    # -----------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------

    reference_year = artifacts.get(
        "reference_year", 2026
    )

    data["Age"] = (
        reference_year - data["Year_Birth"]
    )

    data["Total_Children"] = (
        data["Kidhome"] + data["Teenhome"]
    )

    data["Total_Spending"] = (
        data[SPENDING_COLS].sum(axis=1)
    )

    data["Total_Purchases"] = (
        data[PURCHASE_COLS].sum(axis=1)
    )

    data["Total_Campaign_Acceptance"] = (
        data[CAMPAIGN_COLS].sum(axis=1)
    )

    # -----------------------------------------------------
    # Select exactly the training features and their order
    # -----------------------------------------------------

    features = data.reindex(
        columns=feature_columns
    ).copy()

    for col in features.columns:
        features[col] = pd.to_numeric(
            features[col], errors="coerce"
        )

    if features.isna().any().any():

        invalid_columns = features.columns[
            features.isna().any()
        ].tolist()

        raise ValueError(
            "Missing or invalid values in these model features: "
            + ", ".join(invalid_columns)
            + ". Please correct the uploaded data."
        )

    if not np.isfinite(
        features.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Infinite numeric values found in the dataset."
        )

    return data, features, excluded_count


# =========================================================
# 4. ASSIGN DBSCAN CLUSTERS
# =========================================================

def assign_clusters(features, artifacts):

    scaled_data = artifacts["scaler"].transform(
        features
    )

    pca_data = artifacts["pca"].transform(
        scaled_data
    )

    core_points = np.asarray(
        artifacts["core_points"]
    )

    core_labels = np.asarray(
        artifacts["core_labels"]
    )

    eps = float(artifacts["eps"])

    assigned_labels = []

    for point in pca_data:

        distances = np.linalg.norm(
            core_points - point,
            axis=1
        )

        nearby_indices = np.flatnonzero(
            distances <= eps
        )

        if len(nearby_indices) == 0:
            assigned_labels.append(-1)
            continue

        nearby_labels = core_labels[
            nearby_indices
        ]

        nearby_labels = nearby_labels[
            nearby_labels != -1
        ]

        if len(nearby_labels) == 0:
            assigned_labels.append(-1)
            continue

        unique_labels, counts = np.unique(
            nearby_labels,
            return_counts=True
        )

        assigned_labels.append(
            unique_labels[np.argmax(counts)]
        )

    return np.asarray(assigned_labels)


# =========================================================
# 5. UPLOAD DATASET
# =========================================================

st.sidebar.header("Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload customer dataset",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info(
        "Upload your original marketing campaign Excel or CSV file."
    )
    st.stop()

try:

    if uploaded_file.name.lower().endswith(".csv"):
        raw_data = pd.read_csv(uploaded_file)
    else:
        raw_data = pd.read_excel(uploaded_file)

    raw_data.columns = (
        raw_data.columns.astype(str).str.strip()
    )

except Exception as exc:
    st.error(f"Unable to read uploaded file: {exc}")
    st.stop()


# =========================================================
# 6. DATASET OVERVIEW
# =========================================================

st.header("Uploaded Dataset")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Customer records",
    f"{len(raw_data):,}"
)

col2.metric(
    "Dataset columns",
    f"{len(raw_data.columns):,}"
)

col3.metric(
    "Missing cells",
    f"{int(raw_data.isna().sum().sum()):,}"
)

with st.expander("Preview uploaded data"):
    st.dataframe(
        raw_data.head(10),
        use_container_width=True
    )


# =========================================================
# 7. RUN ANALYSIS
# =========================================================

if st.button(
    "Analyze Customer Behaviour",
    type="primary"
):

    try:

        with st.spinner(
            "Preprocessing customers and assigning clusters..."
        ):

            cleaned_data, features, excluded_count = (
                preprocess_customers(
                    raw_data,
                    artifacts
                )
            )

            labels = assign_clusters(
                features,
                artifacts
            )

            if len(cleaned_data) != len(labels):
                raise ValueError(
                    "Customer records and cluster labels do not align."
                )

            # Preserve only original rows that passed preprocessing.
            results = raw_data.loc[
                cleaned_data.index
            ].copy()

            # Add engineered features for business analysis.
            for col in ENGINEERED_COLS:
                results[col] = cleaned_data[col]

            results["Assigned_Cluster"] = labels

        st.session_state["results"] = results
        st.session_state["excluded_count"] = excluded_count

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.stop()


# Retain results across Streamlit reruns.
results = st.session_state.get("results")

if results is None:
    st.stop()

excluded_count = st.session_state.get(
    "excluded_count", 0
)


# =========================================================
# 8. SEGMENTATION SUMMARY
# =========================================================

st.success("Customer analysis completed successfully.")

clustered = results[
    results["Assigned_Cluster"] != -1
]

noise = results[
    results["Assigned_Cluster"] == -1
]

number_of_clusters = (
    clustered["Assigned_Cluster"].nunique()
)

st.header("Segmentation Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Uploaded customers",
    f"{len(raw_data):,}"
)

c2.metric(
    "Customers analyzed",
    f"{len(results):,}"
)

c3.metric(
    "Clusters found",
    number_of_clusters
)

c4.metric(
    "Noise / unassigned",
    f"{len(noise):,}"
)

if excluded_count:
    st.warning(
        f"{excluded_count} records were excluded because "
        "Education or Marital_Status values were missing or "
        "not present during training."
    )

if len(results):
    noise_percentage = len(noise) / len(results) * 100

    st.metric(
        "Noise percentage",
        f"{noise_percentage:.1f}%"
    )

st.caption(
    "DBSCAN has no native predict() method. New records are "
    "assigned approximately using the saved core points."
)


# =========================================================
# 9. CUSTOMER BEHAVIOUR PROFILES
# =========================================================

st.header("Customer Behaviour Analysis")

profile_columns = [
    "Income",
    "Age",
    "Total_Spending",
    "Total_Purchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Total_Campaign_Acceptance",
    "Recency",
    "Total_Children"
]

profile_columns = [
    col for col in profile_columns
    if col in clustered.columns
]

if not clustered.empty and profile_columns:

    profiles = (
        clustered
        .groupby("Assigned_Cluster")[profile_columns]
        .mean(numeric_only=True)
        .round(2)
    )

    st.subheader("Average Behaviour by Cluster")

    st.write(
        "Use this table to compare spending, purchasing, income, "
        "recency and campaign engagement across customer groups."
    )

    st.dataframe(
        profiles,
        use_container_width=True
    )

    metric_options = [
        col for col in [
            "Total_Spending",
            "Total_Purchases",
            "Income",
            "Total_Campaign_Acceptance",
            "Recency"
        ]
        if col in profiles.columns
    ]

    if metric_options:

        selected_metric = st.selectbox(
            "Select a metric to compare",
            metric_options
        )

        chart_data = (
            profiles[[selected_metric]]
            .reset_index()
        )

        chart = px.bar(
            chart_data,
            x="Assigned_Cluster",
            y=selected_metric,
            title=f"Average {selected_metric} by Cluster",
            text_auto=".2s"
        )

        st.plotly_chart(
            chart,
            use_container_width=True
        )

else:
    st.warning(
        "There are no non-noise customers to compare. "
        "Review the DBSCAN model and its assignment method."
    )


# =========================================================
# 10. PRODUCT SPENDING ANALYSIS
# =========================================================

st.header("Product Spending Behaviour")

available_spending = [
    col for col in SPENDING_COLS
    if col in results.columns
]

if available_spending:

    product_totals = (
        results[available_spending]
        .sum()
        .sort_values(ascending=False)
    )

    spending_data = (
        product_totals
        .rename_axis("Product Category")
        .reset_index(name="Total Spending")
    )

    spending_data["Product Category"] = (
        spending_data["Product Category"]
        .str.replace("Mnt", "", regex=False)
    )

    fig = px.bar(
        spending_data,
        x="Product Category",
        y="Total Spending",
        title="Total Spending by Product Category",
        text_auto=".2s"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig = px.histogram(
        results,
        x="Total_Spending",
        nbins=30,
        title="Distribution of Customer Spending"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# 11. PURCHASE CHANNEL ANALYSIS
# =========================================================

st.header("Purchase Channel Behaviour")

available_channels = [
    col for col in PURCHASE_COLS
    if col in results.columns
]

if available_channels:

    channel_totals = (
        results[available_channels]
        .sum()
        .rename_axis("Channel")
        .reset_index(name="Total Purchases")
    )

    channel_names = {
        "NumWebPurchases": "Web",
        "NumCatalogPurchases": "Catalog",
        "NumStorePurchases": "Store"
    }

    channel_totals["Channel"] = (
        channel_totals["Channel"].map(channel_names)
    )

    fig = px.bar(
        channel_totals,
        x="Channel",
        y="Total Purchases",
        title="Total Purchases by Channel",
        text_auto=".2s"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# 12. MARKETING CAMPAIGN ANALYSIS
# =========================================================

st.header("Marketing Campaign Engagement")

available_campaigns = [
    col for col in CAMPAIGN_COLS
    if col in results.columns
]

if available_campaigns:

    campaign_totals = (
        results[available_campaigns]
        .sum()
        .rename_axis("Campaign")
        .reset_index(name="Accepted Customers")
    )

    fig = px.bar(
        campaign_totals,
        x="Campaign",
        y="Accepted Customers",
        title="Accepted Marketing Offers",
        text_auto=".2s"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# 13. AUTOMATIC BUSINESS INSIGHTS
# =========================================================

st.header("Business Insights")

insights = []

if not clustered.empty:

    # Highest and lowest spending clusters.
    spending_means = (
        clustered
        .groupby("Assigned_Cluster")["Total_Spending"]
        .mean()
        .sort_values(ascending=False)
    )

    if not spending_means.empty:

        high_cluster = spending_means.index[0]
        high_spending = spending_means.iloc[0]

        insights.append(
            f"Cluster {high_cluster} has the highest average "
            f"spending ({high_spending:,.2f}). Consider customer "
            "retention and loyalty strategies for this group."
        )

        low_cluster = spending_means.index[-1]
        low_spending = spending_means.iloc[-1]

        if low_cluster != high_cluster:
            insights.append(
                f"Cluster {low_cluster} has the lowest average "
                f"spending ({low_spending:,.2f}). Investigate its "
                "product preferences and engagement patterns."
            )

    # Purchase frequency.
    purchase_means = (
        clustered
        .groupby("Assigned_Cluster")["Total_Purchases"]
        .mean()
        .sort_values(ascending=False)
    )

    if not purchase_means.empty:
        insights.append(
            f"Cluster {purchase_means.index[0]} has the highest "
            f"average purchase count "
            f"({purchase_means.iloc[0]:.2f})."
        )

    # Campaign acceptance.
    campaign_means = (
        clustered
        .groupby("Assigned_Cluster")[
            "Total_Campaign_Acceptance"
        ]
        .mean()
        .sort_values(ascending=False)
    )

    if not campaign_means.empty:
        insights.append(
            f"Cluster {campaign_means.index[0]} has the highest "
            "average campaign acceptance. Examine which offers "
            "work best for this group."
        )

if len(results):

    noise_percentage = len(noise) / len(results) * 100

    if noise_percentage > 50:
        insights.append(
            f"{noise_percentage:.1f}% of customers are marked "
            "as noise or unassigned. Validate DBSCAN parameters "
            "and cluster assignments before making business decisions."
        )

if insights:
    for insight in insights:
        st.markdown(f"- {insight}")
else:
    st.info(
        "Not enough clustered customers are available "
        "to generate meaningful comparisons."
    )

st.caption(
    "These insights describe observed patterns; they do not "
    "establish the causes of customer behaviour."
)


# =========================================================
# 14. RESULTS AND DOWNLOAD
# =========================================================

st.header("Customer Segmentation Results")

with st.expander("View all customer records"):

    st.dataframe(
        results,
        use_container_width=True
    )

csv_data = results.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Customer Segmentation CSV",
    data=csv_data,
    file_name="customer_segments.csv",
    mime="text/csv"
)

st.divider()

st.caption(
    "Customer Behaviour & Segmentation | Python | Pandas | "
    "Streamlit | DBSCAN"
)
