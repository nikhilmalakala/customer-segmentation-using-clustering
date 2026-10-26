import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import io
import os
import re

# -------------------------------------------------------------
# PAGE CONFIGURATION & METADATA
# -------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Analytics",
    page_icon="📊",
    layout="wide"
)

# Professional analytics dashboard CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(49, 46, 129, 0.15);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #FFFFFF;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #E0E7FF;
        margin-top: 6px;
        font-weight: 400;
    }
    
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 14px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
        position: relative;
        overflow: hidden;
    }
    .metric-card-primary { border-top: 4px solid #4F46E5; }
    .metric-card-success { border-top: 4px solid #10B981; }
    .metric-card-warning { border-top: 4px solid #F59E0B; }
    .metric-card-info    { border-top: 4px solid #06B6D4; }
    .metric-card-purple  { border-top: 4px solid #8B5CF6; }
    .metric-card-rose    { border-top: 4px solid #F43F5E; }
    
    .metric-val {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .metric-lbl {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    
    .badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 6px;
    }
    .badge-num  { background-color: #EEF2FF; color: #4338CA; border: 1px solid #C7D2FE; }
    .badge-cat  { background-color: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
    .badge-id   { background-color: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
    
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 10px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .stDownloadButton > button {
        background-color: #4F46E5;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.8rem;
        border: none;
        transition: all 0.2s ease;
        box-shadow: 0 4px 10px rgba(79, 70, 229, 0.25);
    }
    .stDownloadButton > button:hover {
        background-color: #4338CA;
        color: white;
        box-shadow: 0 6px 14px rgba(79, 70, 229, 0.35);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# HERO HEADER
# -------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">CUSTOMER SEGMENTATION ANALYTICS</div>
    <div class="hero-subtitle">Upload any business dataset to dynamically inspect data quality, select features, and uncover customer segments using K-Means clustering.</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 1. FILE UPLOAD & DYNAMIC INPUT
# -------------------------------------------------------------
st.markdown('<div class="section-header">📁 Upload Dataset</div>', unsafe_allow_html=True)

col_upload, col_sample = st.columns([3, 2])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload your CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload any customer dataset from retail, showroom, e-commerce, supermarket, etc. (CSV, XLSX, XLS supported)"
    )

sample_options = {
    "None (Upload your own)": None,
    "🛍️ Retail Mall Customers (Age, Income, Spending)": "sample_data/sample_customers.csv",
    "🏍️ Bike Showroom (Models, Prices, Service, Visits)": "sample_data/sample_bike_showroom.csv",
    "🛒 E-Commerce Store (Orders, Amounts, Visits, Recency)": "sample_data/sample_ecommerce_store.csv",
    "📊 Retail Customers Excel (.xlsx)": "sample_data/sample_customers.xlsx"
}

with col_sample:
    st.write("")
    selected_sample_label = st.selectbox(
        "Or explore with a sample business dataset:",
        options=list(sample_options.keys()),
        index=0,
        help="Switch between different business datasets to see dynamic adaptation in real-time."
    )

df_raw = None
data_source_name = ""

if uploaded_file is not None:
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        elif filename.endswith((".xlsx", ".xls")):
            df_raw = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a .csv, .xlsx, or .xls file.")
            st.stop()
        data_source_name = uploaded_file.name
    except Exception as e:
        st.error(f"Error reading uploaded file: {str(e)}")
        st.stop()
elif selected_sample_label != "None (Upload your own)" and sample_options[selected_sample_label] is not None:
    sample_file = sample_options[selected_sample_label]
    if os.path.exists(sample_file):
        try:
            if sample_file.endswith(".csv"):
                df_raw = pd.read_csv(sample_file)
            else:
                df_raw = pd.read_excel(sample_file)
            data_source_name = os.path.basename(sample_file)
            st.info(f"Loaded sample dataset: **{selected_sample_label}**")
        except Exception as e:
            st.error(f"Error reading sample dataset: {str(e)}")
            st.stop()
else:
    st.info("👆 Please upload a CSV or Excel dataset, or select one of the sample business datasets above to begin.")
    st.stop()

# Validate that file has data
if df_raw is None or df_raw.empty or len(df_raw) == 0:
    st.error("The uploaded dataset contains no data rows. Please upload a valid dataset.")
    st.stop()

# -------------------------------------------------------------
# 2. DYNAMIC DATASET INSPECTION & QUALITY ANALYSIS
# -------------------------------------------------------------
total_rows = len(df_raw)
total_cols = len(df_raw.columns)
num_cols_all = df_raw.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_all = df_raw.select_dtypes(exclude=[np.number]).columns.tolist()
total_missing = int(df_raw.isnull().sum().sum())
total_duplicates = int(df_raw.duplicated().sum())

# Heuristic Identifier Detection
id_keywords = [
    "id", "identifier", "code", "sno", "sr_no", "srno", "key", "uuid", "guid",
    "cust_id", "customer_id", "customerid", "client_id", "clientid",
    "order_id", "orderid", "transaction_id", "transactionid",
    "invoice_id", "invoiceid", "phone", "mobile", "email", "account_no",
    "accountnumber", "ref_no", "index"
]

def detect_column_role(col_name, series):
    lower_col = col_name.lower().replace(" ", "_").replace("-", "_")
    unique_count = series.nunique(dropna=True)
    unique_ratio = unique_count / max(1, total_rows)
    is_numeric = pd.api.types.is_numeric_dtype(series)
    
    # 1. Check keyword match
    for kw in id_keywords:
        if kw == lower_col or lower_col.startswith(kw + "_") or lower_col.endswith("_" + kw) or ("_" + kw + "_") in lower_col:
            return "Identifier"
            
    # 2. Monotonically increasing or exact 1:1 integer index
    if is_numeric and pd.api.types.is_integer_dtype(series):
        if unique_count == total_rows:
            diffs = series.diff().dropna()
            if len(diffs) > 0 and (diffs == 1).all():
                return "Identifier"
                
    # 3. High uniqueness non-numeric string (e.g. UUID, email, code)
    if not is_numeric and unique_ratio >= 0.95 and total_rows >= 10:
        return "Identifier"
        
    if is_numeric:
        return "Numerical Feature"
    else:
        return "Categorical"

column_roles = {col: detect_column_role(col, df_raw[col]) for col in df_raw.columns}
detected_identifiers = [col for col, role in column_roles.items() if role == "Identifier"]
numerical_candidates = [col for col in num_cols_all if col not in detected_identifiers]
if not numerical_candidates and num_cols_all:
    # If all numerical cols were flagged as identifiers, fall back to all numerical cols
    numerical_candidates = num_cols_all

st.markdown("---")

# Visual Metric Cards
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.markdown(f'<div class="metric-card metric-card-primary"><div class="metric-val">{total_rows:,}</div><div class="metric-lbl">Total Records</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card metric-card-info"><div class="metric-val">{total_cols}</div><div class="metric-lbl">Attributes</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card metric-card-purple"><div class="metric-val">{len(num_cols_all)}</div><div class="metric-lbl">Numerical</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card metric-card-success"><div class="metric-val">{len(cat_cols_all)}</div><div class="metric-lbl">Categorical</div></div>', unsafe_allow_html=True)
with m5:
    st.markdown(f'<div class="metric-card metric-card-warning"><div class="metric-val">{total_missing}</div><div class="metric-lbl">Missing Values</div></div>', unsafe_allow_html=True)
with m6:
    st.markdown(f'<div class="metric-card metric-card-rose"><div class="metric-val">{total_duplicates}</div><div class="metric-lbl">Duplicates</div></div>', unsafe_allow_html=True)

st.write("")

# -------------------------------------------------------------
# 3. DATASET OVERVIEW & QUALITY BREAKDOWN
# -------------------------------------------------------------
st.markdown('<div class="section-header">🔍 Dataset Overview & Column Breakdown</div>', unsafe_allow_html=True)

tab_preview, tab_quality = st.tabs(["📋 Dataset Preview", "🔬 Column Inspection & Data Quality"])

with tab_preview:
    st.dataframe(df_raw.head(10), use_container_width=True)

with tab_quality:
    quality_records = []
    for col in df_raw.columns:
        s = df_raw[col]
        missing_count = int(s.isnull().sum())
        missing_pct = (missing_count / total_rows) * 100
        unique_cnt = int(s.nunique(dropna=True))
        role = column_roles.get(col, "Unknown")
        quality_records.append({
            "Column Name": col,
            "Data Type": str(s.dtype),
            "Non-Null Count": total_rows - missing_count,
            "Missing Values": f"{missing_count} ({missing_pct:.1f}%)",
            "Unique Values": unique_cnt,
            "Detected Role": role
        })
    df_quality = pd.DataFrame(quality_records)
    st.dataframe(df_quality, use_container_width=True)

if detected_identifiers:
    st.caption(f"💡 **Identifier Detection:** Flagged `{', '.join(detected_identifiers)}` as potential identifiers (excluded from default clustering features to avoid noise).")

st.markdown("---")

# -------------------------------------------------------------
# 4. FEATURE SELECTION & CATEGORICAL DATA HANDLING
# -------------------------------------------------------------
st.markdown('<div class="section-header">⚙️ Feature Selection & Preprocessing</div>', unsafe_allow_html=True)

if len(num_cols_all) == 0:
    st.error("The uploaded dataset contains no numerical attributes. Customer segmentation requires numerical features (such as spending, frequency, price, count, etc.).")
    st.stop()

col_feat1, col_feat2 = st.columns(2)

with col_feat1:
    st.write("##### Numerical Features")
    selected_num_features = st.multiselect(
        "Select numerical features for clustering:",
        options=num_cols_all,
        default=numerical_candidates,
        help="Select features representing customer behavior, metrics, or financial values."
    )

with col_feat2:
    st.write("##### Categorical Features (Optional Encoding)")
    # Find low-cardinality non-identifier categorical columns suitable for one-hot encoding
    cat_candidates = [
        c for c in cat_cols_all 
        if c not in detected_identifiers and 1 < df_raw[c].nunique(dropna=True) <= 15
    ]
    if cat_candidates:
        selected_cat_features = st.multiselect(
            "Select categorical features to encode (One-Hot Encoding):",
            options=cat_candidates,
            default=[],
            help="Categorical columns are converted into numeric dummy variables. (Low cardinality ≤15 unique categories recommended)."
        )
    else:
        selected_cat_features = []
        if cat_cols_all:
            st.caption("ℹ️ Categorical columns detected, but none have low cardinality (≤15 unique values) or suitable non-identifier properties.")
        else:
            st.caption("ℹ️ No categorical columns present in this dataset.")

# Data Preprocessing Pipeline
# 1. Deduplicate
df_clean = df_raw.drop_duplicates().copy()
removed_duplicates = total_rows - len(df_clean)

# 2. Build feature matrix
if not selected_num_features and not selected_cat_features:
    st.warning("⚠️ Please select at least 2 features to perform customer clustering.")
    st.stop()

# Prepare numerical subset with median imputation
X_parts = []
feature_names = []

if selected_num_features:
    X_num = df_clean[selected_num_features].copy()
    # Median imputation for numerical missing values
    X_num_imputed = X_num.fillna(X_num.median())
    X_parts.append(X_num_imputed)
    feature_names.extend(selected_num_features)

# Handle categorical encoding if requested
if selected_cat_features:
    df_cat_subset = df_clean[selected_cat_features].astype(str)
    df_cat_encoded = pd.get_dummies(df_cat_subset, drop_first=True, dtype=float)
    X_parts.append(df_cat_encoded)
    feature_names.extend(df_cat_encoded.columns.tolist())

X_combined = pd.concat(X_parts, axis=1)

if X_combined.shape[1] < 2:
    st.warning(f"⚠️ You have selected {X_combined.shape[1]} feature(s). Please select at least 2 features to perform cluster segmentation.")
    st.stop()

if len(df_clean) < 3:
    st.error("The dataset contains too few records for clustering (at least 3 records required).")
    st.stop()

# 3. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

missing_imputed_count = int(df_clean[selected_num_features].isnull().sum().sum()) if selected_num_features else 0

st.markdown("---")

# -------------------------------------------------------------
# 5. CLUSTER ANALYSIS (ELBOW METHOD & DISTRIBUTION)
# -------------------------------------------------------------
st.markdown('<div class="section-header">📈 Cluster Analysis</div>', unsafe_allow_html=True)

# Calculate inertia for k in 2 to min(10, n_samples - 1)
max_k = min(10, len(X_scaled) - 1)

if max_k < 2:
    st.error("Not enough records to perform clustering.")
    st.stop()

k_range = list(range(2, max_k + 1))
inertias = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Geometric point-to-line elbow detection
x1, y1 = float(k_range[0]), float(inertias[0])
x2, y2 = float(k_range[-1]), float(inertias[-1])
dx = x2 - x1
dy = y2 - y1
line_norm = float(np.hypot(dx, dy))

if line_norm > 0:
    distances = [
        abs(dy * float(k_val) - dx * float(in_val) + x2 * y1 - y2 * x1) / line_norm
        for k_val, in_val in zip(k_range, inertias)
    ]
    optimal_k_suggested = int(k_range[np.argmax(distances)])
else:
    optimal_k_suggested = 3

col_elbow, col_distrib = st.columns([1, 1])

with col_elbow:
    st.write("##### Elbow Method (Optimal k)")
    fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4), dpi=130)
    ax_elbow.plot(k_range, inertias, marker='o', color='#4F46E5', linewidth=2.4, markersize=7, label='Inertia')
    ax_elbow.axvline(x=optimal_k_suggested, color='#F59E0B', linestyle='--', linewidth=1.5, label=f'Elbow Suggestion (k={optimal_k_suggested})')
    ax_elbow.set_title("Elbow Method For Optimal k", fontsize=11, fontweight='bold', color='#1E293B', pad=10)
    ax_elbow.set_xlabel("Number of Clusters (k)", fontsize=9, color='#475569')
    ax_elbow.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=9, color='#475569')
    ax_elbow.set_xticks(k_range)
    ax_elbow.grid(True, linestyle='--', alpha=0.4, color='#CBD5E1')
    ax_elbow.legend(loc='best', fontsize=8.5)
    plt.tight_layout()
    st.pyplot(fig_elbow)
    plt.close(fig_elbow)
    
    n_clusters = st.slider(
        "Choose Number of Clusters (k):",
        min_value=2,
        max_value=max_k,
        value=optimal_k_suggested,
        help="Use the elbow graph above to pick the number of customer segments."
    )

# -------------------------------------------------------------
# 6. K-MEANS EXECUTION & CLUSTER LABELS
# -------------------------------------------------------------
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df_clean['Cluster'] = cluster_labels

# Dynamic color palette
palette = sns.color_palette("tab10", n_colors=n_clusters)

with col_distrib:
    st.write("##### Cluster Distribution")
    cluster_counts = df_clean['Cluster'].value_counts().sort_index()
    
    fig_dist, ax_dist = plt.subplots(figsize=(6, 4), dpi=130)
    bars = ax_dist.bar(
        [f"Cluster {c}" for c in cluster_counts.index],
        cluster_counts.values,
        color=palette,
        edgecolor='#1E293B',
        linewidth=0.7
    )
    ax_dist.set_title(f"Customer Distribution Across {n_clusters} Clusters", fontsize=11, fontweight='bold', color='#1E293B', pad=10)
    ax_dist.set_ylabel("Number of Customers", fontsize=9, color='#475569')
    ax_dist.grid(axis='y', linestyle='--', alpha=0.4, color='#CBD5E1')
    
    # Value annotations on bars
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df_clean)) * 100
        ax_dist.annotate(
            f"{int(h)} ({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1E293B'
        )
    plt.tight_layout()
    st.pyplot(fig_dist)
    plt.close(fig_dist)

st.markdown("---")

# -------------------------------------------------------------
# 7. CUSTOMER SEGMENTATION VISUALIZATION
# -------------------------------------------------------------
st.markdown('<div class="section-header">🎯 Customer Segmentation Visualization</div>', unsafe_allow_html=True)

fig_scat, ax_scat = plt.subplots(figsize=(9, 5.2), dpi=130)

if len(selected_num_features) == 2 and not selected_cat_features:
    # Directly plot the 2 chosen numerical features
    feat_x, feat_y = selected_num_features[0], selected_num_features[1]
    for c_idx in range(n_clusters):
        mask = df_clean['Cluster'] == c_idx
        ax_scat.scatter(
            df_clean.loc[mask, feat_x],
            df_clean.loc[mask, feat_y],
            color=palette[c_idx],
            label=f"Cluster {c_idx}",
            alpha=0.8,
            edgecolors='white',
            linewidths=0.5,
            s=55
        )
    # Centroids in original feature scale
    centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    ax_scat.scatter(
        centers_orig[:, 0],
        centers_orig[:, 1],
        c='#0F172A',
        marker='X',
        s=140,
        linewidths=1.8,
        edgecolors='white',
        label='Centroids'
    )
    ax_scat.set_xlabel(feat_x, fontsize=10, fontweight='600', color='#334155')
    ax_scat.set_ylabel(feat_y, fontsize=10, fontweight='600', color='#334155')
    ax_scat.set_title(f"Customer Segments ({feat_x} vs {feat_y})", fontsize=12, fontweight='bold', color='#1E293B', pad=12)
else:
    # Multiple features: use PCA to project onto 2 components for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    centers_pca = pca.transform(kmeans.cluster_centers_)
    exp_var = pca.explained_variance_ratio_ * 100
    
    for c_idx in range(n_clusters):
        mask = df_clean['Cluster'] == c_idx
        ax_scat.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            color=palette[c_idx],
            label=f"Cluster {c_idx}",
            alpha=0.8,
            edgecolors='white',
            linewidths=0.5,
            s=55
        )
    ax_scat.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        c='#0F172A',
        marker='X',
        s=140,
        linewidths=1.8,
        edgecolors='white',
        label='Centroids'
    )
    ax_scat.set_xlabel(f"Principal Component 1 ({exp_var[0]:.1f}% variance)", fontsize=10, fontweight='600', color='#334155')
    ax_scat.set_ylabel(f"Principal Component 2 ({exp_var[1]:.1f}% variance)", fontsize=10, fontweight='600', color='#334155')
    ax_scat.set_title(f"Customer Segments (PCA 2D Projection — {exp_var.sum():.1f}% Total Variance Explained)", fontsize=12, fontweight='bold', color='#1E293B', pad=12)

ax_scat.legend(loc='best', fontsize=9, framealpha=0.9, facecolor='#FFFFFF')
ax_scat.grid(True, linestyle='--', alpha=0.4, color='#CBD5E1')
plt.tight_layout()
st.pyplot(fig_scat)
plt.close(fig_scat)

st.markdown("---")

# -------------------------------------------------------------
# 8. CLUSTER SUMMARY & PROFILES
# -------------------------------------------------------------
st.markdown('<div class="section-header">📊 Cluster Summary & Behavioral Profiles</div>', unsafe_allow_html=True)
st.write("Average metrics across selected features for each customer segment:")

# Compute mean of selected numerical features
if selected_num_features:
    cluster_means = df_clean.groupby('Cluster')[selected_num_features].mean().round(2).reset_index()
    counts_list = df_clean['Cluster'].value_counts().sort_index().values
    cluster_means.insert(1, 'Customer Count', counts_list)
    cluster_means.insert(2, 'Percentage', [f"{(c / len(df_clean)) * 100:.1f}%" for c in counts_list])
    
    # Rename feature columns to Avg <Feature>
    rename_dict = {feat: f"Avg {feat}" for feat in selected_num_features}
    cluster_summary_display = cluster_means.rename(columns=rename_dict)
    cluster_summary_display['Cluster'] = cluster_summary_display['Cluster'].apply(lambda x: f"Cluster {x}")
    st.dataframe(cluster_summary_display, use_container_width=True)
else:
    counts_list = df_clean['Cluster'].value_counts().sort_index().values
    cluster_means = pd.DataFrame({
        "Cluster": [f"Cluster {i}" for i in range(n_clusters)],
        "Customer Count": counts_list,
        "Percentage": [f"{(c / len(df_clean)) * 100:.1f}%" for c in counts_list]
    })
    st.dataframe(cluster_means, use_container_width=True)

st.markdown("---")

# -------------------------------------------------------------
# 9. SEGMENTED DATASET & DOWNLOAD
# -------------------------------------------------------------
st.markdown('<div class="section-header">📑 Segmented Dataset</div>', unsafe_allow_html=True)
st.write("Complete dataset with the assigned `Cluster` column for every customer record:")

st.dataframe(df_clean, use_container_width=True)

# Prepare CSV for download
csv_buffer = io.StringIO()
df_clean.to_csv(csv_buffer, index=False)
csv_bytes = csv_buffer.getvalue().encode('utf-8')

col_dl, col_info = st.columns([1, 2])
with col_dl:
    st.download_button(
        label="📥 Download Segmented Dataset (CSV)",
        data=csv_bytes,
        file_name="segmented_customer_dataset.csv",
        mime="text/csv",
        help="Download the complete dataset including the assigned Cluster column as a CSV file."
    )
with col_info:
    st.caption(f"Export includes **{len(df_clean):,}** records with original columns plus cluster assignments (**Cluster 0** to **Cluster {n_clusters-1}**).")
