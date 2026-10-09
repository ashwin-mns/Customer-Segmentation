# Customer Segmentation Dashboard

An interactive **Streamlit dashboard** for customer segmentation using the **DBSCAN clustering algorithm**. This project analyses customer behaviour, identifies customer groups, and presents useful business insights through visualizations.

## Features

- Upload customer data in CSV or Excel format.
- Perform customer feature engineering.
- Apply feature scaling using StandardScaler.
- Reduce dimensionality using Principal Component Analysis (PCA).
- Assign customers to DBSCAN clusters using saved core points and cluster labels.
- Visualize customer spending behaviour.
- Analyse purchase channels and campaign acceptance.
- View cluster-wise customer behaviour and business insights.
- Download processed customer data as a CSV file.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- DBSCAN Clustering
- Principal Component Analysis (PCA)
- StandardScaler
- Joblib
- Plotly
- OpenPyXL

## Machine Learning Workflow

1. **Data Loading:** Upload the customer marketing campaign dataset.
2. **Data Preprocessing:** Handle missing values and prepare categorical and numerical features.
3. **Feature Engineering:** Create features such as Age, Total Children, Total Spending, Total Purchases, and Total Campaign Acceptance.
4. **Feature Scaling:** Standardize the features using StandardScaler.
5. **Dimensionality Reduction:** Apply PCA to reduce the number of dimensions while retaining important information.
6. **Model Building:** Use DBSCAN to identify customer clusters and noise points.
7. **Cluster Assignment:** Assign uploaded customers using distances to saved core points in PCA space.
8. **Visualization:** Display customer behaviour, cluster summaries, and business insights.

## Project Structure

```text
customer-segmentation-dashboard/
│
├── app.py
├── requirements.txt
├── README.md
│
└── deployment_artifacts/
    └── dbscan_bundle.joblib
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/customer-segmentation-dashboard.git
cd customer-segmentation-dashboard
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## Model Information

**Algorithm:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

**Dimensionality Reduction:** PCA

**Feature Scaling:** StandardScaler

**DBSCAN Parameters:**
- Epsilon (`eps`): 3.5
- Minimum Samples (`min_samples`): 8

DBSCAN groups customers based on density and identifies outliers as noise points, commonly labelled `-1`.

Since DBSCAN does not provide a native `predict()` method, this application uses saved core points and cluster labels to approximate cluster assignments for new customer records.

## Business Insights

The dashboard helps explore:

- Customer spending patterns.
- Purchase behaviour across different channels.
- Campaign acceptance patterns.
- Differences in behaviour between customer clusters.
- Potential noise points and unusual customer records.

These insights can help support customer targeting, marketing analysis, and business decision-making.

## Deployment

This application can be deployed using **Streamlit Community Cloud**.

1. Upload the project files to a GitHub repository.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with GitHub.
4. Create a new application.
5. Select your repository and the `main` branch.
6. Set the main file path to `app.py`.
7. Click **Deploy**.

Ensure that `deployment_artifacts/dbscan_bundle.joblib` is included in the repository and that the application loads it using the correct relative path.

## Important Notes

- Upload a dataset containing the columns expected by the application.
- Do not upload private or sensitive customer data to a public repository.
- Never commit passwords, API keys, access tokens, or secret configuration files.
- Load only trusted serialized model files.
- Ensure that `requirements.txt` contains the dependencies required by the application.

## Author

**Ashwin T.**

Customer Segmentation using Machine Learning and Streamlit.
