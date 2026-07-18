import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# --- 1. CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="Mobile Recommendation Engine", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        background: linear-gradient(135deg, #101828 0%, #1d2939 55%, #344054 100%);
        color: white;
        padding: 1.5rem 1.75rem;
        border-radius: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 50px rgba(16, 24, 40, 0.24);
    }
    .hero-card h1, .hero-card p {
        margin: 0;
    }
    .section-card {
        background: white;
        padding: 1rem 1.1rem;
        border: 1px solid #eaecf0;
        border-radius: 1rem;
        box-shadow: 0 6px 20px rgba(16, 24, 40, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <h1>Mobile Product Segmentation & Recommendation System</h1>
        <p>Fill in a few product preferences and get mobile recommendations that match your budget and feature needs.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 2. LOAD DATASETS ---
@st.cache_data
def load_data():
    dataset_dir = Path('Dataset')
    preferred_file = dataset_dir / 'segmented_mobile_products.xlsx'
    legacy_file = dataset_dir / 'segmented_mobile_products.xls'

    data_file = preferred_file if preferred_file.exists() else legacy_file
    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find {preferred_file.name} or {legacy_file.name} in {dataset_dir}."
        )

    if data_file.suffix.lower() == '.xls':
        with data_file.open('r', encoding='utf-8-sig') as file_handle:
            header_sample = file_handle.read(256)

        if ',' in header_sample and not header_sample.startswith(('PK', '\xD0\xCF')):
            df_original = pd.read_csv(data_file)
            df_features = pd.read_csv(data_file)
        else:
            df_original = pd.read_excel(data_file, engine='xlrd')
            df_features = pd.read_excel(data_file, engine='xlrd')
    else:
        df_original = pd.read_excel(data_file, engine='openpyxl')
        df_features = pd.read_excel(data_file, engine='openpyxl')
    return df_original, df_features

try:
    df, df_scaled = load_data()
except FileNotFoundError:
    st.error("⚠️ Data file not found. Please make sure your Excel file is inside your 'Dataset' folder and named either 'segmented_mobile_products.xlsx' or 'segmented_mobile_products.xls'.")
    st.stop()

numeric_feature_columns = [
    'price_usd',
    'rating',
    'battery_life_rating',
    'camera_rating',
    'performance_rating',
    'design_rating',
    'display_rating',
    'helpful_votes'
]

available_feature_columns = [column for column in numeric_feature_columns if column in df.columns]
feature_ranges = {
    column: (float(df[column].min()), float(df[column].max()))
    for column in available_feature_columns
}

def normalize_series(series, minimum, maximum):
    if maximum == minimum:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - minimum) / (maximum - minimum)

normalized_features = df[available_feature_columns].copy()
for column in available_feature_columns:
    minimum, maximum = feature_ranges[column]
    normalized_features[column] = normalize_series(normalized_features[column], minimum, maximum)

brand_options = ['Any'] + [
    brand for brand in ['Apple', 'Google', 'Motorola', 'OnePlus', 'Realme', 'Samsung', 'Xiaomi']
    if brand in df.columns
]

mobile_name_columns = [
    column for column in [
        'Edge 50', 'Galaxy A55', 'Galaxy Note 20', 'Galaxy S24', 'Galaxy Z Flip',
        'Mi 13 Pro', 'Moto G Power', 'OnePlus 11R', 'OnePlus 12', 'OnePlus Nord 3',
        'Pixel 6', 'Pixel 7a', 'Pixel 8', 'Poco X6', 'Razr 40', 'Realme 12 Pro',
        'Realme Narzo 70', 'Redmi Note 13', 'iPhone 13', 'iPhone 14', 'iPhone 15 Pro',
        'iPhone SE'
    ]
    if column in df.columns
]

def get_mobile_name(row):
    for column in mobile_name_columns:
        if row[column] == 1:
            return column
    return "Unknown Mobile"

df['Mobile Name'] = df.apply(get_mobile_name, axis=1)

cluster_mapping = {
    0: 'Cluster 0: Premium Underperformers',
    1: 'Cluster 1: The Crowd Pleasers',
    2: 'Cluster 2: The Budget Traps'
}
df['Cluster_Name'] = df['Cluster'].map(cluster_mapping)

# --- 3. FEATURE-BASED RECOMMENDATION FORM ---
st.sidebar.header("Your Preferences")

with st.sidebar.form("recommendation_form"):
    budget_value = st.slider(
        "Budget (USD)",
        min_value=float(df['price_usd'].min()),
        max_value=float(df['price_usd'].max()),
        value=float(df['price_usd'].median()),
        step=1.0,
    )
    rating_value = st.slider("Minimum Rating", 1.0, 5.0, 4.0, 0.1)
    battery_value = st.slider("Battery Life", 1.0, 5.0, 3.5, 0.1)
    camera_value = st.slider("Camera", 1.0, 5.0, 4.0, 0.1)
    performance_value = st.slider("Performance", 1.0, 5.0, 4.0, 0.1)
    design_value = st.slider("Design", 1.0, 5.0, 3.5, 0.1)
    display_value = st.slider("Display", 1.0, 5.0, 3.5, 0.1)
    brand_value = st.selectbox("Preferred Brand", brand_options)
    top_n = st.slider("Number of Recommendations", min_value=3, max_value=10, value=5)
    submitted = st.form_submit_button("Find Recommendations")

if submitted:
    target_vector = pd.DataFrame([
        {
            'price_usd': budget_value,
            'rating': rating_value,
            'battery_life_rating': battery_value,
            'camera_rating': camera_value,
            'performance_rating': performance_value,
            'design_rating': design_value,
            'display_rating': display_value,
            'helpful_votes': float(df['helpful_votes'].median()) if 'helpful_votes' in df.columns else 0.0,
        }
    ])

    target_vector = target_vector.reindex(columns=available_feature_columns, fill_value=0.0)
    normalized_target = target_vector.copy()
    for column in available_feature_columns:
        minimum, maximum = feature_ranges[column]
        normalized_target[column] = normalize_series(normalized_target[column], minimum, maximum)

    similarities = cosine_similarity(normalized_target, normalized_features).flatten()
    recommendations = df.copy()
    recommendations['Similarity Score'] = similarities

    if brand_value != 'Any' and brand_value in recommendations.columns:
        recommendations = recommendations[recommendations[brand_value] == 1]

    recommendations = recommendations.sort_values('Similarity Score', ascending=False).head(top_n)

    st.markdown("### Recommended Products")
    st.caption("These matches are ranked by similarity to the preferences you filled in.")

    if recommendations.empty:
        st.warning("No products matched the current filters. Try relaxing the brand preference or adjusting the feature sliders.")
    else:
        top_pick = recommendations.iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Match", top_pick['Mobile Name'])
        col2.metric("Best Match Price", f"${top_pick['price_usd']:.2f}")
        col3.metric("Best Match Similarity", f"{top_pick['Similarity Score']:.3f}")

        display_columns = [
            column for column in ['Mobile Name', 'price_usd', 'rating', 'battery_life_rating', 'camera_rating', 'performance_rating', 'design_rating', 'display_rating', 'Similarity Score']
            if column in recommendations.columns
        ]

        st.dataframe(
            recommendations[display_columns].reset_index(drop=True).style.format({
                'price_usd': '${:.2f}',
                'rating': '{:.1f}',
                'battery_life_rating': '{:.1f}',
                'camera_rating': '{:.1f}',
                'performance_rating': '{:.1f}',
                'design_rating': '{:.1f}',
                'display_rating': '{:.1f}',
                'Similarity Score': '{:.4f}'
            }),
            use_container_width=True,
        )

        st.markdown("### Why this works")
        explanation_col1, explanation_col2 = st.columns(2)
        explanation_col1.markdown(
            f"""
            <div class="section-card">
                <strong>Matched profile</strong><br>
                Budget around ${budget_value:.0f}, rating at least {rating_value:.1f}, and balanced feature scores.
            </div>
            """,
            unsafe_allow_html=True,
        )
        explanation_col2.markdown(
            f"""
            <div class="section-card">
                <strong>Preference filter</strong><br>
                Brand preference: {brand_value}. Recommendations are ranked by feature similarity.
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
st.subheader("📊 Product Segment Overview")

cluster_feature_columns = [
    column for column in ['price_usd', 'rating', 'battery_life_rating', 'camera_rating', 'performance_rating', 'design_rating', 'display_rating']
    if column in df.columns
]

cluster_summary = df.groupby('Cluster_Name').agg(
    Product_Count=('Cluster', 'size'),
    Average_Price=('price_usd', 'mean'),
    Average_Rating=('rating', 'mean'),
    Average_Camera=('camera_rating', 'mean'),
    Average_Performance=('performance_rating', 'mean')
).reset_index()

tab1, tab2, tab3, tab4 = st.tabs([
    "Cluster Size",
    "Price vs Rating",
    "Feature Heatmap",
    "Radar Profiles"
])

with tab1:
    count_order = df['Cluster_Name'].value_counts().index.tolist()
    fig_count = px.bar(
        cluster_summary.sort_values('Product_Count', ascending=False),
        x='Cluster_Name',
        y='Product_Count',
        color='Cluster_Name',
        title='Products per Cluster',
        labels={'Cluster_Name': 'Cluster', 'Product_Count': 'Product Count'},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_count.update_layout(showlegend=False)
    st.plotly_chart(fig_count, use_container_width=True)
    st.caption('This shows how the dataset is distributed across your three segments.')

with tab2:
    fig_scatter = px.scatter(
        df.sample(min(len(df), 4000), random_state=7),
        x='price_usd',
        y='rating',
        color='Cluster_Name',
        title='Price vs Rating by Cluster',
        labels={'price_usd': 'Price (USD)', 'rating': 'User Rating'},
        hover_data=['camera_rating', 'performance_rating', 'Mobile Name'],
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption('This helps you see whether a cluster is premium, budget, or balanced in the market.')

with tab3:
    heatmap_columns = [column for column in cluster_feature_columns if column in df.columns and column != 'price_usd']
    cluster_heatmap = df.groupby('Cluster_Name')[heatmap_columns].mean().reset_index()
    heatmap_data = cluster_heatmap.set_index('Cluster_Name')
    fig_heatmap = px.imshow(
        heatmap_data,
        text_auto='.2f',
        color_continuous_scale='Blues',
        aspect='auto',
        title='Average Feature Score by Cluster',
    )
    fig_heatmap.update_layout(xaxis_title='Feature', yaxis_title='Cluster')
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.caption('The heatmap makes it easy to compare which cluster is stronger on each feature.')

with tab4:
    radar_features = [column for column in ['rating', 'battery_life_rating', 'camera_rating', 'performance_rating', 'design_rating', 'display_rating'] if column in df.columns]
    radar_profile = df.groupby('Cluster_Name')[radar_features].mean().reset_index()

    radar_fig = go.Figure()
    for _, row in radar_profile.iterrows():
        radar_fig.add_trace(
            go.Scatterpolar(
                r=row[radar_features].tolist() + [row[radar_features].tolist()[0]],
                theta=radar_features + [radar_features[0]],
                fill='toself',
                name=row['Cluster_Name'],
            )
        )

    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        title='Cluster Feature Profiles',
        showlegend=True,
    )
    st.plotly_chart(radar_fig, use_container_width=True)
    st.caption('Radar profiles are useful when you want to compare clusters across multiple feature dimensions at once.')