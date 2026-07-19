# 📱 Mobile Product Segmentation and Recommendation System

An end-to-end data analytics and machine learning pipeline that segments mobile phone products into market clusters (premium, mid-range, budget-style groups) and recommends similar products using a content-based, similarity-driven approach — wrapped in an interactive Streamlit application.

---

## 1. Problem Statement

The smartphone industry generates large volumes of structured review data — specifications, pricing, ratings, and user information — that is often too unorganized to analyze directly. This project cleans and transforms that raw data, applies clustering to uncover meaningful product segments, and builds a recommendation engine that suggests similar phones based on product features.

## 2. Objective

Follow the complete data lifecycle:

`Data Collection → Cleaning & Preprocessing → EDA → Feature Engineering & Scaling → Clustering → Cluster Analysis → Recommendation System → Streamlit Application`

## 3. Dataset

- **Source file:** `Dataset/Mobile Reviews Sentiment null.csv`
- **Size:** 50,000 reviews × 22 raw columns
- **Fields:** review id, customer name, age, brand, model, price (USD/local + exchange rate), rating, sentiment, country, language, review date, verified purchase flag, sub-ratings (battery, camera, performance, design, display), helpful votes, source (platform)
- **Brands (7):** Apple, Google, Motorola, OnePlus, Realme, Samsung, Xiaomi
- **Countries (8):** Australia, Brazil, Canada, Germany, India, UAE, UK, USA
- **Models (22):** e.g. iPhone 13/14/15 Pro/SE, Galaxy S24/A55/Note 20/Z Flip, Pixel 6/7a/8, OnePlus 11R/12/Nord 3, Realme 12 Pro/Narzo 70, Redmi Note 13, Poco X6, Mi 13 Pro, Moto G Power, Razr 40, Edge 50

## 4. Project Structure

```text
Mobile Product Segmentation and Recommendation System Using Python and Machine Learning/
├── app.py                                  # Streamlit application (recommendations + cluster dashboards)
├── requirements.txt                        # Python dependencies
├── README.md
├── Dataset/
│   ├── Mobile Reviews Sentiment null.csv   # Raw dataset (50,000 rows)
│   ├── Processed_data.xls                  # Cleaned & encoded dataset (CSV content, 56 cols)
│   ├── Scaled_data.xls                     # Standard-scaled feature matrix (CSV content, 56 cols)
│   └── segmented_mobile_products.xls       # Final dataset with cluster labels (CSV content, 57 cols)
└── Notebook/
    ├── Data Cleaning.ipynb                 # Steps 1–3: collection, cleaning, preprocessing, EDA
    └── Model.ipynb                         # Steps 4–5: clustering, cluster analysis, recommendation logic
```

> **Note:** The three files in `Dataset/` carry an `.xls` extension but actually contain plain CSV text (a naming artifact from the export step). `pandas.read_csv()` / the app's loader read them correctly, but renaming them to `.csv` would make the project easier to maintain.

## 5. Methodology

### Step 1 — Data Collection
The raw CSV is loaded into a Pandas DataFrame and inspected with `.head()`, `.shape`, `.info()`, and `.describe()` to understand structure and types (`Data Cleaning.ipynb`).

### Step 2 — Data Cleaning & Preprocessing
- **Missing values:** `price_usd` (2,450 nulls) and `rating` (2,453 nulls) are imputed with the **median per phone model**; `sentiment` (2,445 nulls) and `source` (2,448 nulls) are filled with `"Unknown"`.
- **Duplicates:** checked with `.duplicated().sum()` — **0 duplicate rows found**, so no rows were dropped.
- **Irrelevant/redundant columns removed:** `review_id`, `customer_name`, `price_local`, `currency`, `exchange_rate_to_usd`, `review_date`, `language`, `age` (identifiers or values already represented by `price_usd`).
- **Type conversion:** `verified_purchase` and `rating` cast to `int`.
- **Categorical encoding:** `model`, `brand`, `sentiment`, `country`, `source` one-hot encoded via `pd.get_dummies`.
- **Feature scaling:** all 56 resulting features standardized with `StandardScaler` (zero mean, unit variance) and saved as `Scaled_data.csv`.

### Step 3 — Exploratory Data Analysis
- Product/review distribution by **brand** and **country** (bar charts).
- **Top 5 / bottom 5** rated phone models.
- **Price vs. rating** scatter plot and **price vs. camera rating** boxplot.
- **Correlation heatmap** across price, rating, and all sub-ratings.
- Statistical summaries (mean, std, min, max) of price/rating/performance split by sentiment class.

### Step 4 — Clustering (Segmentation)
- **Algorithm:** K-Means (`sklearn.cluster.KMeans`, `init='k-means++'`, `random_state=42`).
- **Choosing K:** rather than fixing K at the brief's illustrative value of 4, the number of clusters was determined empirically — an **elbow plot (WCSS)** was generated for K = 1–6, and **silhouette scores** were compared for K = 2 (0.0755) and K = 3 (0.0772). K = 3 was selected as the best-supported split for this dataset.
- **Result:** 3 clusters — 23,443 / 19,505 / 7,052 products — saved to `segmented_mobile_products.csv`.

| Cluster | Persona | Avg. Price | Avg. Rating | Avg. Camera | Avg. Performance |
|---|---|---|---|---|---|
| 1 | **The Crowd Pleasers** (largest, high value) | $655.79 | 3.99 | 3.52 | 3.52 |
| 0 | **Premium Underperformers** (smallest, overpriced) | $900.58 | 3.11 | 2.70 | 2.70 |
| 2 | **The Budget Traps** (mid-priced, poor quality) | $654.46 | 2.07 | 1.76 | 1.76 |

**Key insight:** Clusters 1 and 2 cost almost the same (~$655) yet Cluster 1 nearly doubles Cluster 2's satisfaction scores — price alone is a poor predictor of quality in this dataset. Cluster 0 sits in its own high-price bracket without a matching quality premium.

### Step 5 — Recommendation System
A **content-based, similarity-driven** recommender is implemented in two complementary forms:
1. **`Model.ipynb`** — given a product's index, `get_recommendations()` restricts the candidate pool to the product's own cluster, then ranks peers by **cosine similarity** on the scaled feature vector — validated by inspecting a sample product against its top-5 matches.
2. **`app.py`** — a live, user-facing version: the user's slider inputs (budget, rating, battery, camera, performance, design, display) are min-max normalized and compared via cosine similarity against every normalized product in the dataset, with optional brand filtering.

No collaborative filtering / user-user history is used — recommendations are purely feature-based, as intended by the brief.

### Step 6 — Streamlit Application
`app.py` provides:
- A **preference form** (sidebar sliders + brand dropdown + result count) that returns ranked recommendations with similarity scores.
- A **"Why this works"** explanation panel for the top match.
- Four visualization tabs built with **Plotly**: Cluster Size, Price vs. Rating, Feature Heatmap, and Radar Profiles.

## 6. Tech Stack

Python · Pandas · NumPy · scikit-learn (KMeans, StandardScaler, cosine_similarity, silhouette_score) · Matplotlib · Seaborn · Plotly · Streamlit

## 7. Installation

```bash
git clone <your-repo-url>
cd "Mobile Product Segmentation and Recommendation System Using Python and Machine Learning"
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 8. Running the App

```bash
streamlit run app.py
```
Then open the local URL Streamlit prints (defaults to `http://localhost:8501`).

## 9. How to Use

1. Open the app in your browser.
2. Set your desired budget, minimum rating, and feature-quality sliders in the sidebar.
3. (Optional) pick a preferred brand.
4. Click **Find Recommendations**.
5. Review the ranked phone list, similarity scores, and the top match summary.
6. Explore the **Product Segment Overview** tabs to understand how clusters differ.

## 10. Task Completion Summary

| Project requirement | Status |
|---|---|
| CSV → DataFrame conversion | ✅ `Data Cleaning.ipynb` |
| Missing value handling | ✅ group-wise median / "Unknown" fill |
| Duplicate removal | ✅ checked — none present |
| Feature selection & type conversion | ✅ |
| Categorical encoding (brand, country, model, etc.) | ✅ one-hot via `get_dummies` |
| Feature scaling | ✅ `StandardScaler` |
| EDA (distributions, top/bottom products, correlations, stats) | ✅ |
| K-Means clustering with justified cluster count | ✅ K=3 (elbow + silhouette) |
| Cluster interpretation / business personas | ✅ 3 named segments |
| Similarity-based recommendation system | ✅ cosine similarity (cluster-based + preference-based) |
| Visualizations (Plotly/Seaborn/Matplotlib) | ✅ |
| Interactive Streamlit app | ✅ verified running without errors |
| Documentation | ✅ this README |

**Overall:** the project covers every stage of the brief's pipeline. The only deliberate deviation is the cluster count (K=3 instead of the brief's illustrative K=4) — the brief explicitly labels 4 as "for example only," and the choice here is backed by an elbow plot and silhouette comparison, which is arguably stronger evidence than fixing K in advance.

## 11. Known Limitations / Suggested Polish

- The dataset files in `Dataset/` are CSV content saved with an `.xls` extension — works today, but renaming to `.csv` would avoid confusion.
- `requirements.txt` doesn't include `matplotlib`/`seaborn`, which the notebooks (not the app) depend on — worth adding for full reproducibility.
- Preprocessing and recommendation logic live in notebooks plus `app.py` rather than separate standalone `.py` scripts — fine for this format, but could be refactored into a `src/` module if the project grows.
- Recommendation relevance is validated with a spot-check example rather than a formal metric (e.g., average intra-cluster similarity) — a nice-to-have enhancement, not a gap in the brief.

## 12. Future Improvements

- Add a trained/learned recommendation model as an alternative to pure similarity ranking.
- Side-by-side phone comparison view.
- Search and sort by brand or price in the results table.
- Persist user feedback to refine future recommendations.

## Author
Puneeth Raja
