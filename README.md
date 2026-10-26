# Customer Segmentation Analytics Using Clustering

An end-to-end, enterprise-ready Machine Learning web application built with Python and Streamlit that performs customer segmentation using K-Means clustering across arbitrary business datasets.

## Key Highlights

- **Universal Dataset Support**: Accepts any CSV (`.csv`) or Excel (`.xlsx`, `.xls`) customer dataset without hardcoded column names (e.g. retail malls, bike/car showrooms, supermarkets, e-commerce, electronics, apparel).
- **Dynamic Dataset Inspection**: Automatically extracts dataset dimensions, attribute types, non-null counts, missing value rates, unique counts, and duplicate rows.
- **Intelligent Identifier Detection**: Uses pattern recognition and statistical heuristics (e.g. `CustomerID`, `Order_ID`, high-uniqueness tokens, sequential indices) to flag and exclude identifiers from default clustering.
- **Categorical Data Handling**: Identifies categorical features, explains non-numeric constraints for Euclidean distance clustering, and offers optional one-hot encoding for suitable low-cardinality attributes.
- **Data Preprocessing Pipeline**: Automatically deduplicates data, applies median imputation for missing numerical values, and standardizes variables using `StandardScaler`.
- **Dynamic Feature Selection**: Automatically populates checkboxes/multiselect with actual column names from the uploaded dataset.
- **Elbow Method & Optimization**: Computes K-Means inertia across $k \in [2, 10]$ and detects the optimal elbow using geometric perpendicular distance to guide cluster count selection.
- **K-Means Clustering**: Fits `KMeans(n_clusters=k, random_state=42, n_init=10)` and assigns cluster labels to every usable customer record.
- **Modern Analytical Dashboard**:
  - **KPI Metrics Cards**: Total Records, Attributes, Numerical Count, Categorical Count, Missing Values, Duplicates.
  - **Dataset Overview & Column Quality Breakdown**: Full preview and column-level quality matrix.
  - **Elbow Method Curve & Cluster Distribution Bar Chart**: Side-by-side analytical graphs with exact customer counts and percentage annotations.
  - **2D Customer Segmentation Visualization**: Direct 2D feature scatter plot (for 2 features) or 2D PCA projection (for $>2$ features) displaying cluster separation, variance explained, and cluster centroids.
  - **Cluster Summary & Profiles**: Table presenting mean behavioral values and demographics per cluster segment.
  - **Segmented Dataset Export**: Instant preview and one-click CSV download containing original data plus assigned cluster labels.

## Project Structure

```text
├── app.py                            # Main Streamlit web application
├── requirements.txt                  # Application dependencies
├── README.md                         # Project documentation
└── sample_data/
    ├── sample_customers.csv          # Retail mall customer dataset (Age, Income, Spending)
    ├── sample_customers.xlsx         # Retail mall customer dataset (Excel format)
    ├── sample_bike_showroom.csv      # Bike showroom dataset (Models, Prices, Service, Visits)
    └── sample_ecommerce_store.csv    # E-commerce dataset (Orders, Amounts, Visits, Recency)
```

## Getting Started

### 1. Prerequisites

- Python 3.10+ installed

### 2. Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .\.venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will automatically open in your default browser at:
```text
http://localhost:8501
```

### 4. Testing with Sample Data

You can upload your own business CSV/Excel file or choose from any of the built-in business datasets via the dropdown:
- **Retail Mall Customers**: Demonstrates consumer spending and income segmentation.
- **Bike Showroom Customers**: Demonstrates showroom pricing, service count, visit volume, and bike model categorical encoding.
- **E-Commerce Store Customers**: Demonstrates transaction value, order volume, visit counts, and purchase recency.
